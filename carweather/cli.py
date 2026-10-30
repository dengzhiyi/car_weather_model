# -*- coding: utf-8 -*-
"""命令行入口：predict / correct / sample 三个子命令。

约定
----
* stdout 只输出 JSON（便于车载服务或管道消费）；
* 所有人类可读文本（提示、摘要、警告）一律走 stderr；
* ``--out`` 把同一份 JSON 落盘，与 stdout 完全一致。
"""

from __future__ import annotations

import argparse
import json
import sys

from .common import parse_dt
from .pipeline import correct, predict
from .sources import make_sample


def _summary(obj):
    """把结果压成 4 行人读摘要（stderr）。"""
    m, f, c, fi = obj["meta"], obj["flood"], obj["camping"], obj["fishing"]
    stage = {"pre_departure": "出发前", "post_arrival": "抵达实测"}[m["stage"]]
    print(f"[{stage}] {m['dest'][0]:.3f},{m['dest'][1]:.3f}  ETA {m['eta']}  "
          f"数据 {m['data_source']}  置信度 {m['confidence']:.2f}  "
          f"高程 {m['elevation_m']}m 坡度 {m['slope_deg']}°", file=sys.stderr)
    print(f"  山洪 {f['level']}级({f['label']}) "
          f"概率 {['%.0f%%' % (x * 100) for x in f['prob']]} "
          f"峰值 {f['peak_time']} 触发 {f['onset_time'] or '无'}", file=sys.stderr)
    print(f"  露营 {c['score']:.0f}分 {c['verdict']} 窗口 {c['window'] or '无'} "
          f"夜间最低 {c['min_temp']}℃ 结露 {c['condensation_hours']}h", file=sys.stderr)
    print(f"  钓鱼 {fi['score']:.0f}分 最佳 {fi['best_time']} "
          f"窗口 {fi['windows'] or '无'}", file=sys.stderr)
    if "delta" in obj:
        print(f"  相对出发前变化 {obj['delta']}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="车载气象三任务（山洪/露营/钓鱼）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="出发前：目的地纯预报推理")
    p1.add_argument("--lat", type=float, required=True)
    p1.add_argument("--lon", type=float, required=True)
    p1.add_argument("--eta", required=True)
    p1.add_argument("--stay-hours", type=float, default=12)
    p1.add_argument("--offline", action="store_true")
    p1.add_argument("--out")
    p1.add_argument("--summary", action="store_true")

    p2 = sub.add_parser("correct", help="抵达驻车后：实测校正")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--lat", type=float, required=True)
    p2.add_argument("--lon", type=float, required=True)
    p2.add_argument("--eta")
    p2.add_argument("--stay-hours", type=float, default=12)
    p2.add_argument("--offline", action="store_true")
    p2.add_argument("--out")
    p2.add_argument("--summary", action="store_true")

    p3 = sub.add_parser("sample", help="生成一份驻车传感器样例 CSV")
    p3.add_argument("--out", default="samples/sensor_parked.csv")
    p3.add_argument("--minutes", type=int, default=100)
    p3.add_argument("--lat", type=float, default=30.25)
    p3.add_argument("--lon", type=float, default=119.72)
    p3.add_argument("--offline", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "sample":
        make_sample(a.out, a.lat, a.lon, minutes=a.minutes, offline=a.offline)
        return 0

    if a.cmd == "predict":
        obj = predict(a.lat, a.lon, parse_dt(a.eta),
                      stay_hours=a.stay_hours, offline=a.offline)
    else:
        obj = correct(a.csv, a.lat, a.lon, parse_dt(a.eta) if a.eta else None,
                      stay_hours=a.stay_hours, offline=a.offline)

    text = json.dumps(obj, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"已写入 {a.out}", file=sys.stderr)
    if a.summary:
        _summary(obj)
    return 0
