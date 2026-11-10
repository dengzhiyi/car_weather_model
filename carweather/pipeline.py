# -*- coding: utf-8 -*-
"""两段式主流程：出发前预报推理（predict）→ 抵达驻车后实测校正（correct）。

两段式设计的物理动机
--------------------
1. 出发前只能依赖数值预报，山洪这类对小尺度对流极敏感的指标误差最大；
2. 抵达后车辆驻车，5 个传感器（温度/湿度/气压/雨量/光线）进入稳定状态，
   此时做一次单点最优插值（OI）把预报“拉回”实测，并据此重算三任务；
3. 两段输出结构同构（meta/flood/camping/fishing/actions），
   HMI 只需切换 stage 标识，无需改渲染逻辑。
"""

from __future__ import annotations

import datetime as dt

from .astro import solar_position
from .common import UTC, iso_local, nearest_index, rnd
from .contract import (OI_QC_PENALTY, OI_WEIGHT_MAX, OI_WEIGHT_MIN, PARKED_SPEED_KMH,
                       PREHEAT_MIN, STAY_HOURS_DEFAULT)
from .models import actions, camping_model, confidence, fishing_model, flood_model
from .physics import dew_point, ineichen_clear_sky, linke_turbidity_estimate, lux_to_ghi
from .sources import get_elevation, get_slope, get_thresholds, load_sensor_csv, load_weather


# --------------------------------------------------------------------------
# 传感器观测处理
# --------------------------------------------------------------------------

def _is_parked(r) -> bool:
    """驻车判定：显式 mode=parked，或未给 mode 但速度 < 3 km/h。"""
    return r["mode"] == "parked" or (r["mode"] == "" and r["speed"] < PARKED_SPEED_KMH)


def aggregate_obs(rows, lat: float, lon: float, elevation_m: float = 50.0):
    """把驻车传感器时序聚合成一个“观测时刻”的代表值。

    处理链：
      1. 只保留驻车样本（硬约束：行驶中数据不参与任何计算）；
      2. 丢弃驻车后前 5 min 预热数据；
      3. 逐项量程 QC，记录 flags；
      4. 光线与晴空理论值比对，低于 25% 视为隧道/树荫/车灯遮挡，剔除；
      5. 最后 10 个样本取均值（抗单点噪声），雨量按最近 1 h 累加。
    """
    parked = [r for r in rows if _is_parked(r)]
    if not parked:
        raise ValueError("没有可用的驻车样本（传感器仅在驻车状态参与计算）")

    start = parked[0]["dt"] + dt.timedelta(minutes=PREHEAT_MIN)
    valid = [r for r in parked if r["dt"] >= start]
    if not valid:
        raise ValueError("驻车样本不足 5 分钟预热期")

    flags, keep = [], []
    for r in valid:
        bad = []
        if not (-40 <= r["t"] <= 60):
            bad.append("t_range")
        if not (0 <= r["rh"] <= 100):
            bad.append("rh_range")
        if not (800 <= r["p"] <= 1080):
            bad.append("p_range")
        if not (0 <= r["rain"] <= 30):
            bad.append("rain_range")
        if not (0 <= r["lux"] <= 130000):
            bad.append("lux_range")
        if bad:
            flags.extend(bad)
        else:
            keep.append(r)
    if not keep:
        raise ValueError("全部样本 QC 未通过")

    # 光线校验：与 Ineichen 晴空辐照比对（夜间或太阳高度 <10° 不判云）
    lux_vals, elevs = [], []
    doy = keep[-1]["dt"].timetuple().tm_yday
    tl = linke_turbidity_estimate(lat, doy, elevation_m, 70.0)
    for r in keep[-10:]:
        elev, _ = solar_position(r["dt"], lat, lon)
        elevs.append(elev)
        if elev < 10:
            continue
        clear_ghi = ineichen_clear_sky(elev, elevation_m, tl, doy)[0]
        if clear_ghi > 0 and lux_to_ghi(r["lux"]) < 0.25 * clear_ghi:
            flags.append("lux_occluded")
            continue
        lux_vals.append(r["lux"])

    last10 = keep[-10:]
    span_min = (keep[-1]["dt"] - keep[0]["dt"]).total_seconds() / 60.0
    rain_1h = sum(r["rain"] for r in keep
                  if (keep[-1]["dt"] - r["dt"]).total_seconds() <= 3600)

    # 云量反演：1 − 晴空指数 kt；kt 用近 10 个有效光照样本均值
    cloud_obs = None
    if lux_vals:
        elev = elevs[-1]
        clear_ghi = ineichen_clear_sky(elev, elevation_m, tl, doy)[0]
        if clear_ghi > 50:
            mean_ghi = lux_to_ghi(sum(lux_vals) / len(lux_vals))
            kt = min(1.15, mean_ghi / clear_ghi)
            cloud_obs = max(0.0, min(1.0, 1.0 - kt))

    def mean(key):
        return sum(r[key] for r in last10) / len(last10)

    return {
        "time": keep[-1]["dt"],
        "t": mean("t"),
        "rh": mean("rh"),
        "p": mean("p"),
        "rain_1h": rain_1h,
        "lux": (sum(lux_vals) / len(lux_vals)) if lux_vals else None,
        "cloud": cloud_obs,
        "span_min": span_min,
        "qc_flags": sorted(set(flags)),
        "n_valid": len(keep),
    }


def assimilate(w, idx: int, obs, weight: float):
    """单点最优插值（OI）：X = X_fcst + w·(X_obs − X_fcst)。

    只改当前小时这一列，后续小时仍沿用预报的演变趋势；
    降水取 max(预报, 0.6·实测 1h 雨量)，避免同化把正在下的雨“抹平”。
    """
    wc = {k: (list(v) if isinstance(v, list) else v) for k, v in w.items()}
    i = idx
    wc["temperature"][i] += weight * (obs["t"] - wc["temperature"][i])
    wc["humidity"][i] += weight * (obs["rh"] - wc["humidity"][i])
    wc["dewpoint"][i] = dew_point(wc["temperature"][i], wc["humidity"][i])
    wc["precipitation"][i] = max(wc["precipitation"][i], obs["rain_1h"] * 0.6)
    if obs.get("cloud") is not None:
        wc["cloud"][i] = weight * obs["cloud"] * 100.0 + (1.0 - weight) * wc["cloud"][i]
    return wc


# --------------------------------------------------------------------------
# 推理入口
# --------------------------------------------------------------------------

def predict(lat: float, lon: float, eta: dt.datetime, stay_hours: float = STAY_HOURS_DEFAULT,
            offline: bool = False):
    """出发前：纯预报推理。"""
    w = load_weather(lat, lon, offline=offline)
    elevation = get_elevation(lat, lon, offline=offline)
    slope = get_slope(lat, lon, offline=offline)
    thresholds = get_thresholds(lat, lon, offline=offline)
    idx = nearest_index(w["times"], eta)

    flood = flood_model(w, idx, slope, thresholds)
    camp = camping_model(w, idx, stay_hours)
    fish = fishing_model(w, idx, lat, lon)
    lead_h = (eta - dt.datetime.now(UTC)).total_seconds() / 3600.0

    return {
        "meta": {
            "stage": "pre_departure",
            "generated_at": iso_local(dt.datetime.now(UTC)),
            "dest": [lat, lon],
            "eta": iso_local(eta),
            "stay_hours": stay_hours,
            "data_source": w["source"],
            "elevation_m": rnd(elevation, 0),
            "slope_deg": rnd(slope, 1),
            "confidence": rnd(confidence(lead_h, w["source"], "pre_departure"), 2),
            "needs_recheck_before": iso_local(eta - dt.timedelta(hours=3)),
        },
        "flood": flood,
        "camping": camp,
        "fishing": fish,
        "actions": actions(flood, camp, fish),
    }


def correct(csv_path: str, lat: float, lon: float, eta: dt.datetime | None = None,
            stay_hours: float = STAY_HOURS_DEFAULT, offline: bool = False):
    """抵达驻车后：实测同化 + 三任务重算，并输出与出发前的 delta。"""
    rows = load_sensor_csv(csv_path)
    elevation = get_elevation(lat, lon, offline=offline)
    obs = aggregate_obs(rows, lat, lon, elevation_m=elevation)

    arrival = eta or obs["time"]
    base = predict(lat, lon, arrival, stay_hours=stay_hours, offline=offline)

    w = load_weather(lat, lon, offline=offline)
    idx = nearest_index(w["times"], arrival)
    # QC 问题越多，越不信任实测（权重下限 0.4，上限 0.75）
    weight = max(OI_WEIGHT_MIN, OI_WEIGHT_MAX - OI_QC_PENALTY * len(obs["qc_flags"]))
    wc = assimilate(w, idx, obs, weight)

    slope = get_slope(lat, lon, offline=offline)
    thresholds = get_thresholds(lat, lon, offline=offline)
    flood = flood_model(wc, idx, slope, thresholds)
    camp = camping_model(wc, idx, stay_hours)
    fish = fishing_model(wc, idx, lat, lon)

    instant_bias = {
        "t": rnd(obs["t"] - w["temperature"][idx], 2),
        "rh": rnd(obs["rh"] - w["humidity"][idx], 1),
        "p": rnd(obs["p"] - w["pressure_sta"][idx], 1),
    }

    out = {
        "meta": dict(base["meta"]),
        "flood": flood,
        "camping": camp,
        "fishing": fish,
        "actions": actions(flood, camp, fish),
    }
    out["meta"]["stage"] = "post_arrival"
    out["meta"]["eta"] = iso_local(arrival)
    out["meta"]["confidence"] = rnd(confidence(
        (arrival - dt.datetime.now(UTC)).total_seconds() / 3600.0,
        w["source"], "post_arrival"), 2)
    out["meta"]["data_source"] = w["source"] + "+sensor"

    out["observed"] = {
        "time": iso_local(obs["time"]), "t": rnd(obs["t"], 1), "rh": rnd(obs["rh"], 0),
        "p_sta": rnd(obs["p"], 1), "rain_1h": rnd(obs["rain_1h"], 1),
        "lux": rnd(obs["lux"], 0), "cloud_from_lux": rnd(obs["cloud"], 2),
        "span_min": rnd(obs["span_min"], 0), "n_valid": obs["n_valid"],
        "qc_flags": obs["qc_flags"],
    }
    out["corrections"] = {
        "assimilation_weight": rnd(weight, 2),
        "instant_bias_vs_forecast": instant_bias,
        "note": "instant_bias 仅供入库做长期校准，不直接回灌本次结果",
    }
    out["delta"] = {
        "flood_level": flood["level"] - base["flood"]["level"],
        "camping_score": camp["score"] - base["camping"]["score"],
        "fishing_score": fish["score"] - base["fishing"]["score"],
    }
    out["baseline"] = {
        "flood_level": base["flood"]["level"],
        "camping_score": base["camping"]["score"],
        "fishing_score": base["fishing"]["score"],
    }
    return out
