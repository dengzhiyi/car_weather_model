# -*- coding: utf-8 -*-
"""数据接入层：Open-Meteo（预报 / 历史 / 高程）与本地传感器 CSV。

设计要点
--------
* 所有外网请求都带进程内缓存（key 含四舍五入坐标），一次预测只发 3 个请求：
  预报、高程 9 点、历史降水（可选）。
* 网络失败一律降级到确定性合成数据，并在结果里标注 ``data_source=synthetic``，
  保证车机无网时功能可用、且结果可复现（同一地点同一时刻结果不变）。
* 私有 API 不向上层泄漏：上层只看到统一结构的 ``WeatherSeries`` 字典。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request

from .astro import solar_position
from .common import UTC, nearest_index, parse_dt, prefix_sums, pct, to_float_list, window_sum
from .contract import (ARCHIVE_DAYS, DEFAULT_THRESHOLDS, ENDPOINTS, FORECAST_DAYS,
                       HOURLY_FIELD_NAMES, PAST_DAYS, THRESHOLD_FLOORS,
                       THRESHOLD_PERCENTILES)
from .physics import (dew_point, ghi_to_lux, ineichen_clear_sky, linke_turbidity_estimate)

# 需要的逐时字段（顺序即 URL 顺序，便于日志排查）；字段清单见 contract.py
HOURLY_FIELDS = ",".join(HOURLY_FIELD_NAMES)

_CACHE = {}

# 可选磁盘缓存：仅当显式设置 CARWEATHER_CACHE_DIR 时启用。
# 车机可指向持久分区，避免每次冷启动都重拉 365 天 ERA5 历史。
_CACHE_DIR = os.environ.get("CARWEATHER_CACHE_DIR", "").strip()
_THRESHOLD_TTL_S = 30 * 24 * 3600


def _disk_path(kind: str, lat: float, lon: float):
    if not _CACHE_DIR:
        return None
    return os.path.join(_CACHE_DIR, f"{kind}_{lat:.2f}_{lon:.2f}.json")


def _disk_get(path, ttl: float):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if time.time() - float(record.get("at", 0)) > ttl:
            return None
        return record.get("value")
    except (OSError, ValueError, TypeError):
        return None


def _disk_put(path, value):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"at": time.time(), "value": value}, f)
    except OSError:
        pass


# --------------------------------------------------------------------------
# HTTP 与缓存
# --------------------------------------------------------------------------

def http_json(url: str, timeout: float = 25.0):
    """GET 并解析 JSON；统一 UA，便于服务端限流识别。"""
    req = urllib.request.Request(url, headers={"User-Agent": "car-weather/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_get(key):
    return _CACHE.get(key)


def _cache_put(key, value):
    _CACHE[key] = value
    return value


# --------------------------------------------------------------------------
# Open-Meteo 预报
# --------------------------------------------------------------------------

def fetch_open_meteo(lat: float, lon: float):
    """拉取目的地逐时预报（过去 1 天 + 未来 3 天），返回统一结构。"""
    url = ENDPOINTS["forecast"] + "?" + urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "hourly": HOURLY_FIELDS,
        "daily": "sunrise,sunset", "timezone": "UTC",
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
    })
    j = http_json(url)
    h = j["hourly"]
    times = [dt.datetime.fromisoformat(x).replace(tzinfo=UTC) for x in h["time"]]
    d = j.get("daily", {})
    return {
        "times": times,
        "temperature": to_float_list(h.get("temperature_2m"), 15.0),
        "humidity": to_float_list(h.get("relative_humidity_2m"), 70.0),
        "dewpoint": to_float_list(h.get("dew_point_2m"), 10.0),
        "precipitation": to_float_list(h.get("precipitation")),
        "precip_prob": to_float_list(h.get("precipitation_probability")),
        "pressure_msl": to_float_list(h.get("pressure_msl"), 1013.0),
        "pressure_sta": to_float_list(h.get("surface_pressure"), 1000.0),
        "wind": to_float_list(h.get("wind_speed_10m"), 2.0),
        "gust": to_float_list(h.get("wind_gusts_10m"), 3.0),
        "wind_dir": to_float_list(h.get("wind_direction_10m"), 0.0),
        "cloud": to_float_list(h.get("cloud_cover"), 50.0),
        "radiation": to_float_list(h.get("shortwave_radiation")),
        "uv": to_float_list(h.get("uv_index")),
        "cape": to_float_list(h.get("cape")),
        "soil_moisture": to_float_list(h.get("soil_moisture_0_to_1cm"), 0.3),
        "visibility": to_float_list(h.get("visibility"), 15000.0),
        "sunrise": [dt.datetime.fromisoformat(x).replace(tzinfo=UTC) for x in d.get("sunrise", [])],
        "sunset": [dt.datetime.fromisoformat(x).replace(tzinfo=UTC) for x in d.get("sunset", [])],
        "elevation": float(j.get("elevation") or 0.0),
        "source": "open-meteo",
    }


def synthetic_weather(lat: float, lon: float):
    """离线合成数据：确定性、含一次降水过程，字段与 Open-Meteo 完全同构。

    用于无网络演示与回归测试。光照用 Ineichen 晴空值乘云衰减，
    保证 ``lux`` 与 ``radiation`` 一致（云量反演才能闭环）。
    """
    now = dt.datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - dt.timedelta(hours=24)
    n = 96
    times = [start + dt.timedelta(hours=i) for i in range(n)]
    base = 20.0 - 0.35 * abs(lat - 25.0)
    elevation_m = 50.0

    arrays = {k: [] for k in (
        "temperature", "humidity", "dewpoint", "precipitation", "precip_prob",
        "pressure_msl", "pressure_sta", "wind", "gust", "wind_dir", "cloud",
        "radiation", "uv", "cape", "soil_moisture", "visibility")}

    for i, t in enumerate(times):
        local_hour = t.astimezone(dt.timezone(dt.timedelta(hours=8))).hour
        tair = base + 4.5 * math.sin(math.radians((local_hour - 9) * 15)) + 1.2 * math.sin(i / 13.0)
        rain = 0.0
        if 8 <= i < 11:
            rain = 0.9                    # 过去一场小雨（抬高 API）
        if 38 <= i < 41:
            rain = 2.4                    # 未来一场中雨
        if 44 <= i < 46:
            rain = 0.6
        prob = 0.85 if rain > 1.5 else (0.5 if rain > 0 else 0.12)
        cloud = 0.85 if rain > 0 else (0.45 if (i % 7) < 3 else 0.2)
        wind = 2.0 + 1.6 * math.sin(i / 9.0)

        elev, _ = solar_position(t, lat, lon)
        tl = linke_turbidity_estimate(lat, t.timetuple().tm_yday, elevation_m, 70.0)
        ghi, _, _ = ineichen_clear_sky(elev, elevation_m, tl, t.timetuple().tm_yday)
        sw = ghi * (1 - 0.65 * cloud)

        arrays["temperature"].append(tair)
        arrays["humidity"].append(min(98.0, 62.0 + 30.0 * (rain / 2.4) + 6.0 * math.sin(i / 11.0)))
        arrays["precipitation"].append(rain)
        arrays["precip_prob"].append(prob)
        pmsl = 1013.0 + 2.8 * math.sin(i / 26.0) - 1.2 * (rain > 1.0)
        arrays["pressure_msl"].append(pmsl)
        arrays["pressure_sta"].append(pmsl - 3.5)
        arrays["wind"].append(wind)
        arrays["gust"].append(wind * 1.9)
        arrays["wind_dir"].append(135.0)
        arrays["cloud"].append(cloud * 100.0)
        arrays["radiation"].append(max(0.0, sw))
        arrays["uv"].append(max(0.0, 7.0 * math.sin(math.radians(max(0.0, elev))) ** 2 * (1 - cloud)))
        arrays["cape"].append(1100.0 if rain > 1.5 else 0.0)
        arrays["soil_moisture"].append(0.31)
        arrays["visibility"].append(12000.0 if rain == 0 else 6000.0)

    arrays["dewpoint"] = [dew_point(arrays["temperature"][i], arrays["humidity"][i]) for i in range(n)]

    return dict(arrays, times=times, sunrise=[], sunset=[],
                elevation=elevation_m, source="synthetic")


def load_weather(lat: float, lon: float, offline: bool = False):
    """统一入口：在线预报，失败或显式离线时用合成数据。"""
    key = ("weather", round(lat, 3), round(lon, 3), offline)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    if offline:
        return _cache_put(key, synthetic_weather(lat, lon))
    try:
        return _cache_put(key, fetch_open_meteo(lat, lon))
    except Exception as e:                    # 网络/接口异常 → 降级
        print(f"[warn] Open-Meteo 不可用，切换合成数据: {e}", file=sys.stderr)
        return _cache_put(key, synthetic_weather(lat, lon))


# --------------------------------------------------------------------------
# 地形
# --------------------------------------------------------------------------

def get_elevation(lat: float, lon: float, offline: bool = False) -> float:
    """目的地海拔（m）。"""
    key = ("elev", round(lat, 3), round(lon, 3), offline)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    if offline:
        return _cache_put(key, 50.0)
    try:
        j = http_json(ENDPOINTS["elevation"] + "?" + urllib.parse.urlencode(
            {"latitude": lat, "longitude": lon}))
        return _cache_put(key, float(j["elevation"][0]))
    except Exception:
        return _cache_put(key, 50.0)


def get_slope(lat: float, lon: float, offline: bool = False) -> float:
    """9 点高程差分估坡度（度），替代 DEM 文件。

    取 3×3 网格（步长 0.01°），中心差分求东西/南北梯度，再取模长。
    一次请求最多 9 个点（Open-Meteo 高程接口支持批量）。
    """
    key = ("slope", round(lat, 2), round(lon, 2), offline)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    if offline:
        return _cache_put(key, 8.0)

    d = 0.01
    lats, lons = [], []
    for r in (1, 0, -1):              # 北 → 南
        for c in (-1, 0, 1):          # 西 → 东
            lats.append(lat + r * d)
            lons.append(lon + c * d)
    try:
        q = urllib.parse.urlencode({
            "latitude": ",".join(f"{x:.4f}" for x in lats),
            "longitude": ",".join(f"{x:.4f}" for x in lons),
        })
        e = [float(x) for x in http_json(ENDPOINTS["elevation"] + "?" + q)["elevation"]]
    except Exception:
        return _cache_put(key, 8.0)

    dx = 2 * d * 111320.0 * math.cos(math.radians(lat))
    dy = 2 * d * 111320.0
    gx = ((e[0] + e[3] + e[6]) - (e[2] + e[5] + e[8])) / 3.0 / dx
    gy = ((e[0] + e[1] + e[2]) - (e[6] + e[7] + e[8])) / 3.0 / dy
    return _cache_put(key, math.degrees(math.atan(math.hypot(gx, gy))))


# --------------------------------------------------------------------------
# 临界雨量（ERA5 历史分位）
# --------------------------------------------------------------------------

def get_thresholds(lat: float, lon: float, offline: bool = False):
    """用 Archive(ERA5) 近一年逐时降水推本地临界雨量。

    思路：取各窗口长度（1/3/6/24 h）滑动累积雨量的 98/95 分位作为“当地极端值”，
    再与经验下限取大，避免干旱区出现“1 mm 就报警”。
    """
    key = ("th", round(lat, 2), round(lon, 2), offline)
    hit = _cache_get(key)
    if hit is not None:
        return hit

    cache_file = _disk_path("thresholds", lat, lon)
    hit = _disk_get(cache_file, _THRESHOLD_TTL_S)
    if hit is not None:
        return _cache_put(key, hit)

    if offline:
        return _cache_put(key, dict(DEFAULT_THRESHOLDS, source="default"))

    try:
        end = dt.date.today() - dt.timedelta(days=1)
        start = end - dt.timedelta(days=ARCHIVE_DAYS)
        url = ENDPOINTS["archive"] + "?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "precipitation", "timezone": "UTC",
        })
        precip = to_float_list(http_json(url, timeout=40)["hourly"]["precipitation"])
    except Exception:
        return _cache_put(key, dict(DEFAULT_THRESHOLDS, source="default"))

    ps = prefix_sums(precip)          # 前缀和：4 个窗口长度一次遍历即可
    n = len(precip)
    cum = {h: [window_sum(ps, i, h) for i in range(n)] for h in (1, 3, 6, 24)}
    th = {
        k: max(THRESHOLD_FLOORS[k], pct(cum[k], THRESHOLD_PERCENTILES[k]))
        for k in ("1h", "3h", "6h", "24h")
    }
    th["source"] = "archive-era5"
    _disk_put(cache_file, th)
    return _cache_put(key, th)


# --------------------------------------------------------------------------
# 本地传感器
# --------------------------------------------------------------------------

def load_sensor_csv(path: str):
    """读驻车传感器 CSV（列：time,mode,t_c,rh_pct,p_sta_hpa,rain_mm,lux,speed_kmh）。

    坏行跳过而不抛异常：车机日志经常残缺。读取后按时间排序。
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "dt": parse_dt(r["time"]),
                    "mode": (r.get("mode") or "").strip().lower(),
                    "t": float(r["t_c"]),
                    "rh": float(r["rh_pct"]),
                    "p": float(r["p_sta_hpa"]),
                    "rain": float(r["rain_mm"]),
                    "lux": float(r["lux"]),
                    "speed": float(r["speed_kmh"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x["dt"])
    return rows


# --------------------------------------------------------------------------
# 样本生成（演示预测 → 校正闭环）
# --------------------------------------------------------------------------

def make_sample(path: str, lat: float = 30.25, lon: float = 119.72,
                minutes: int = 100, offline: bool = False):
    """用目的地实时预报插值生成一份“驻车实测”样例 CSV。

    故意注入 +0.6 ℃ 系统偏差与传感器噪声，用于演示同化权重与偏差统计。
    """
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).replace(second=0, microsecond=0)
    start = now - dt.timedelta(minutes=minutes)
    rng = random.Random(20261030)
    w = load_weather(lat, lon, offline=offline)

    lines = ["time,mode,t_c,rh_pct,p_sta_hpa,rain_mm,lux,speed_kmh"]
    for i in range(minutes + 1):
        ts = start + dt.timedelta(minutes=i)
        tu = ts.astimezone(UTC)
        j = nearest_index(w["times"], tu)
        j2 = min(j + 1, len(w["times"]) - 1)
        f = (tu - w["times"][j]).total_seconds() / 3600.0
        t = w["temperature"][j] * (1 - f) + w["temperature"][j2] * f
        rh = w["humidity"][j] * (1 - f) + w["humidity"][j2] * f
        p = w["pressure_sta"][j] * (1 - f) + w["pressure_sta"][j2] * f
        rad = w["radiation"][j] * (1 - f) + w["radiation"][j2] * f
        rain = w["precipitation"][j] / 60.0

        # 传感器噪声 + 轻微系统偏差（模拟未校准的温度虚高）
        t += 0.6 + 0.06 * (rng.random() - 0.5)
        rh = min(99.5, rh + 0.5 * (rng.random() - 0.5))
        p += 0.02 * (rng.random() - 0.5)
        rain = max(0.0, rain + 0.002 * (rng.random() - 0.5))
        lux = ghi_to_lux(rad) * (0.9 + 0.05 * (rng.random() - 0.5))

        lines.append(f"{ts.isoformat(timespec='seconds')},parked,{t:.2f},{rh:.1f},"
                     f"{p:.2f},{rain:.3f},{lux:.0f},0.0")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"已生成 {path}（{minutes} 分钟驻车样本，含噪声与 +0.6℃ 系统偏差；"
          f"数据源 {w['source']}）", file=sys.stderr)
