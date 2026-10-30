# -*- coding: utf-8 -*-
"""通用工具：时区、数值修约、时间窗求和、分位数与正态 CDF。

这里是全包唯一的“基础层”，不含任何业务语义，便于单测与被其他模块复用。
"""

from __future__ import annotations

import datetime as dt
import math

# 输出统一使用 +08:00（车载 HMI 常用时区），内部计算一律 UTC。
UTC = dt.timezone.utc
LOCAL = dt.timezone(dt.timedelta(hours=8))


# --------------------------------------------------------------------------
# 数值与时间
# --------------------------------------------------------------------------

def rnd(x, n=1):
    """安全修约；None 原样返回，非数值时原样返回（用于 JSON 输出）。"""
    if x is None:
        return None
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def floor_hour(t: dt.datetime) -> dt.datetime:
    """向下取整到整点（保留时区）。"""
    return t.replace(minute=0, second=0, microsecond=0)


def parse_dt(s: str) -> dt.datetime:
    """解析 ISO 时间；无时区按本地时区(+08:00)解释，统一转 UTC。"""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL)
    return d.astimezone(UTC)


def iso_local(t: dt.datetime) -> str:
    """UTC 时间 → 本地 ISO 字符串（分钟精度）。"""
    return t.astimezone(LOCAL).isoformat(timespec="minutes")


def to_float_list(arr, default=0.0):
    """把可能含 None 的序列统一成 float 列表（Open-Meteo 常返回 null）。"""
    return [default if x is None else float(x) for x in (arr or [])]


# --------------------------------------------------------------------------
# 窗口求和：前缀和实现，把 O(n·k) 降到 O(n)
# --------------------------------------------------------------------------

def prefix_sums(arr):
    """返回长度 len(arr)+1 的前缀和，ps[0]=0。

    之后任意闭区间 [lo, hi] 的和为 ``ps[hi+1] - ps[lo]``，
    对 1h/3h/6h/24h 滑动窗口只需一次减法，避免在模型内层反复切片求和。
    """
    ps = [0.0] * (len(arr) + 1)
    acc = 0.0
    for i, x in enumerate(arr):
        acc += x
        ps[i + 1] = acc
    return ps


def window_sum(ps, end, length):
    """前缀和数组上取以 end 结尾、长 length 的窗口和；左边界自动截断。"""
    lo = max(0, end - length + 1)
    return ps[end + 1] - ps[lo]


def pct(values, q):
    """线性插值分位数（q ∈ [0,1]）。"""
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return v[int(k)]
    return v[f] * (c - k) + v[c] * (k - f)


def norm_cdf(x):
    """标准正态 CDF（用 erf 实现，免依赖 scipy）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def nearest_index(times, when):
    """在有序时间序列中找与 when 最接近的下标（线性扫描，序列很短）。"""
    best, bi = None, 0
    for i, t in enumerate(times):
        d = abs((t - when).total_seconds())
        if best is None or d < best:
            best, bi = d, i
    return bi
