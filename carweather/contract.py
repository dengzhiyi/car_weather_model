# -*- coding: utf-8 -*-
"""跨仓库契约：字段、单位、时间口径与窗口参数。

本文件是模型仓与车机前端之间唯一的机读真源，前端仓库保存一份生成的
``contract.json`` 镜像，并用脚本比对，避免两边字段/单位/参数漂移。

用法::

    python3 -m carweather.contract --out contract.json
    python3 -m carweather.contract --check contract.json
"""

from __future__ import annotations

import argparse
import json
import sys

# 契约版本：字段、单位或语义发生不兼容变化时递增
SCHEMA_VERSION = "1.0.0"

# 内部计算一律 UTC；对外展示与传感器日志使用 +08:00
INTERNAL_TZ = "UTC"
DISPLAY_TZ = "+08:00"

# Open-Meteo 三个只读端点（前端直连时需加入 CSP connect-src / 原生白名单）
ENDPOINTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "elevation": "https://api.open-meteo.com/v1/elevation",
}

# 预报窗口：过去 1 天用于 24h 气压/雨量趋势，未来 3 天覆盖 48h 决策视界
PAST_DAYS = 1
FORECAST_DAYS = 3

# (字段名, 单位, 合理下限, 合理上限)；null 表示接口可能返回 null
HOURLY_FIELDS = (
    ("temperature_2m", "°C", -60.0, 60.0),
    ("relative_humidity_2m", "%", 0.0, 100.0),
    ("dew_point_2m", "°C", -90.0, 60.0),
    ("precipitation", "mm", 0.0, 500.0),
    ("precipitation_probability", "%", 0.0, 100.0),
    ("pressure_msl", "hPa", 800.0, 1100.0),
    ("surface_pressure", "hPa", 500.0, 1100.0),
    ("wind_speed_10m", "m/s", 0.0, 120.0),
    ("wind_gusts_10m", "m/s", 0.0, 150.0),
    ("wind_direction_10m", "°", 0.0, 360.0),
    ("cloud_cover", "%", 0.0, 100.0),
    ("shortwave_radiation", "W/m²", 0.0, 1500.0),
    ("uv_index", "1", 0.0, 20.0),
    ("cape", "J/kg", 0.0, 10000.0),
    ("soil_moisture_0_to_1cm", "m³/m³", 0.0, 1.0),
    ("visibility", "m", 0.0, 100000.0),
)

HOURLY_FIELD_NAMES = tuple(f[0] for f in HOURLY_FIELDS)

# 传感器 CSV / 内存行契约（车机驻车日志与前端适配层共用）
SENSOR_COLUMNS = (
    ("time", "ISO8601 " + DISPLAY_TZ, None, None),
    ("mode", "parked|driving|empty", None, None),
    ("t_c", "°C", -40.0, 60.0),
    ("rh_pct", "%", 0.0, 100.0),
    ("p_sta_hpa", "hPa", 800.0, 1080.0),
    ("rain_mm", "mm/采样", 0.0, 30.0),
    ("lux", "lx", 0.0, 130000.0),
    ("speed_kmh", "km/h", 0.0, 250.0),
)

# 驻车判定与热平衡预热
PARKED_SPEED_KMH = 3.0
PREHEAT_MIN = 5.0

# 单点最优插值（OI）权重
OI_WEIGHT_MIN = 0.4
OI_WEIGHT_MAX = 0.75
OI_QC_PENALTY = 0.12

# 决策窗口
STAY_HOURS_DEFAULT = 12.0
FLOOD_HORIZON_H = 48
FISHING_HORIZON_H = 48

# 无历史数据时的临界雨量初值（mm）
DEFAULT_THRESHOLDS = {"1h": 12.0, "3h": 24.0, "6h": 34.0, "24h": 64.0}
# 经验下限（mm），避免干旱区出现“1 mm 就报警”
THRESHOLD_FLOORS = {"1h": 8.0, "3h": 15.0, "6h": 20.0, "24h": 40.0}
# 历史分位：短窗口 98%，24h 95%
THRESHOLD_PERCENTILES = {"1h": 0.98, "3h": 0.98, "6h": 0.98, "24h": 0.95}
ARCHIVE_DAYS = 365

# UTCI 官方适用域（输入裁剪）
UTCI_DOMAIN = {
    "tdb": [-50.0, 50.0],
    "dtr": [-30.0, 70.0],
    "v10": [0.5, 17.0],
    "rh": [0.0, 100.0],
}


def _columns(rows):
    return [
        {"name": name, "unit": unit, "min": lo, "max": hi, "nullable": True}
        for name, unit, lo, hi in rows
    ]


def as_dict():
    """返回完整契约字典（键顺序固定，便于比对）。"""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "timezone": {
            "internal": INTERNAL_TZ,
            "display": DISPLAY_TZ,
            "sensorLog": DISPLAY_TZ,
        },
        "endpoints": dict(ENDPOINTS),
        "hourly": {
            "request": {
                "pastDays": PAST_DAYS,
                "forecastDays": FORECAST_DAYS,
                "cellSelection": "nearest",
            },
            "fields": _columns(HOURLY_FIELDS),
        },
        "sensor": {
            "columns": _columns(SENSOR_COLUMNS),
            "parked": {
                "modeValues": ["parked"],
                "speedMaxKmh": PARKED_SPEED_KMH,
                "preheatMin": PREHEAT_MIN,
            },
            "qc": {
                "ranges": {name: [lo, hi] for name, _unit, lo, hi in SENSOR_COLUMNS
                           if lo is not None},
            },
        },
        "decision": {
            "stayHoursDefault": STAY_HOURS_DEFAULT,
            "floodHorizonH": FLOOD_HORIZON_H,
            "fishingHorizonH": FISHING_HORIZON_H,
            "oi": {
                "weightMin": OI_WEIGHT_MIN,
                "weightMax": OI_WEIGHT_MAX,
                "qcPenalty": OI_QC_PENALTY,
            },
        },
        "thresholds": {
            "defaults": dict(DEFAULT_THRESHOLDS),
            "floors": dict(THRESHOLD_FLOORS),
            "percentiles": dict(THRESHOLD_PERCENTILES),
            "archiveDays": ARCHIVE_DAYS,
            "sourceValues": ["archive-era5", "default"],
        },
        "utci": {"domain": {k: list(v) for k, v in UTCI_DOMAIN.items()}},
        "output": {
            "stages": ["pre_departure", "post_arrival"],
            "keys": ["meta", "flood", "camping", "fishing", "actions"],
            "correctExtraKeys": ["observed", "corrections", "delta", "baseline"],
        },
    }


def dumps():
    return json.dumps(as_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="输出/校验 carweather 跨仓库契约")
    parser.add_argument("--out", help="写入 JSON 契约文件")
    parser.add_argument("--check", help="与给定 JSON 文件比对，不一致时退出码 1")
    args = parser.parse_args(argv)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(dumps())
        print(f"contract written: {args.out}", file=sys.stderr)
        return 0

    if args.check:
        with open(args.check, encoding="utf-8") as f:
            actual = json.load(f)
        expected = as_dict()
        if actual != expected:
            print("contract drift detected", file=sys.stderr)
            for key in sorted(set(actual) | set(expected)):
                if actual.get(key) != expected.get(key):
                    print(f"  - {key}", file=sys.stderr)
            return 1
        print("contract ok", file=sys.stderr)
        return 0

    sys.stdout.write(dumps())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
