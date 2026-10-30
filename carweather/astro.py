# -*- coding: utf-8 -*-
"""天文计算：太阳位置、太阳升落、月球位置 / 升落 / 中天、Solunar 窗口。

实现说明与精度
--------------
* 太阳：NOAA 低精度公式（Cooper/Fourier 近似），赤纬与均时差误差 < 0.5°，
  用于太阳高度、方位、升落时刻（±2 min）。
* 月球：Meeus《Astronomical Algorithms》第 47 章低精度解析式（主要周期项），
  黄经误差约 0.3°，地心视赤经/赤纬可直接用于 1 小时级窗口。
* 恒星时：Meeus 12.4 式（GMST），再叠加经度得到地方恒星时 LST。
* 升落判据：月球取 h0 = +0.125°（同时计入大气折射 34′ 与地平视差 −16′ 的合成效应）；
  太阳取 h0 = −0.833°（折射 34′ + 半径 16′）。

性能取向
--------
* 所有三角函数只在“每 10 分钟一次”的采样点上计算，升落/中天用线性插值收敛，
  避免 Meeus 的整点迭代法（其内部每次迭代都要重算月球位置）。
* Solunar 窗口按 (日期, 经度) 缓存，钓鱼模型逐小时取值时不再重复采样。
"""

from __future__ import annotations

import datetime as dt
import math

from .common import UTC

# 每小时地球自转角度（度）
_DEG_PER_HOUR = 15.0

# 月球升落的地平高度阈值：折射 34′ ≈ 0.5667°，视差 −16′ ≈ −0.2667°。
_MOON_H0 = 0.125
# 太阳升落阈值：−34′ 折射 −16′ 半径。
_SUN_H0 = -0.833

# 太阳/月球通用常数
_SIN = math.sin
_COS = math.cos


# --------------------------------------------------------------------------
# 儒略日 / 恒星时
# --------------------------------------------------------------------------

def julian_day(when: dt.datetime) -> float:
    """UTC 时刻 → 儒略日（JDN + 日内小数）。"""
    y, m = when.year, when.month
    d = when.day + (when.hour + when.minute / 60.0 + when.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def gmst_deg(jd: float) -> float:
    """格林尼治平恒星时（度），Meeus 12.4。"""
    t = (jd - 2451545.0) / 36525.0
    theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * t * t - t * t * t / 38710000.0)
    return theta % 360.0


def local_sidereal_deg(when: dt.datetime, lon: float) -> float:
    """地方恒星时（度），东经为正。"""
    return (gmst_deg(julian_day(when)) + lon) % 360.0


# --------------------------------------------------------------------------
# 太阳
# --------------------------------------------------------------------------

def _solar_terms(when: dt.datetime):
    """返回 (赤纬弧度, 均时差分钟)。NOAA 分数年角近似。"""
    n = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0
    g = 2 * math.pi / 365.0 * (n - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * _COS(g) - 0.032077 * _SIN(g)
                       - 0.014615 * _COS(2 * g) - 0.040849 * _SIN(2 * g))
    decl = (0.006918 - 0.399912 * _COS(g) + 0.070257 * _SIN(g)
            - 0.006758 * _COS(2 * g) + 0.000907 * _SIN(2 * g)
            - 0.002697 * _COS(3 * g) + 0.00148 * _SIN(3 * g))
    return decl, eqtime


def solar_position(when: dt.datetime, lat: float, lon: float):
    """太阳高度角与方位角（度）。

    方位角以北为 0°、顺时针增大（车载朝向/遮阳计算常用约定）。
    """
    decl, eqtime = _solar_terms(when)
    hour = when.hour + when.minute / 60.0
    # 真太阳时（分钟）→ 时角（度）
    tst = hour * 60 + eqtime + 4 * lon
    ha = math.radians(tst / 4.0 - 180.0)

    latr = math.radians(lat)
    sin_lat, cos_lat = _SIN(latr), _COS(latr)
    cos_zen = (sin_lat * _SIN(decl) + cos_lat * _COS(decl) * _COS(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    elevation = 90.0 - math.degrees(math.acos(cos_zen))

    # 由南向方位角转北向：atan2(sinH, cosH sinφ − tanδ cosφ) + 180°
    az_south = math.degrees(math.atan2(_SIN(ha),
                                       _COS(ha) * sin_lat - math.tan(decl) * cos_lat))
    azimuth = (az_south + 180.0) % 360.0
    return elevation, azimuth


def sun_times(date_utc: dt.datetime, lat: float, lon: float):
    """当日（UTC 日期）日出/日落时刻；极昼极夜返回 (None, None)。"""
    # 以当地正午附近的赤纬作为当日代表值
    noon = dt.datetime(date_utc.year, date_utc.month, date_utc.day, 12, tzinfo=UTC) \
        - dt.timedelta(hours=lon / _DEG_PER_HOUR)
    decl, eqtime = _solar_terms(noon)
    latr, dec = math.radians(lat), decl
    # sin(h0) = sinφ sinδ + cosφ cosδ cosH  →  cosH = [sin(h0) − sinφ sinδ] / (cosφ cosδ)
    cos_h0 = (_SIN(math.radians(_SUN_H0)) - _SIN(latr) * _SIN(dec)) \
        / (_COS(latr) * _COS(dec))
    if cos_h0 > 1.0 or cos_h0 < -1.0:
        return None, None
    h0 = math.degrees(math.acos(cos_h0))

    base = dt.datetime(date_utc.year, date_utc.month, date_utc.day, tzinfo=UTC)
    solar_noon = base + dt.timedelta(hours=12.0 - lon / _DEG_PER_HOUR - eqtime / 60.0)
    return (solar_noon - dt.timedelta(hours=h0 / _DEG_PER_HOUR),
            solar_noon + dt.timedelta(hours=h0 / _DEG_PER_HOUR))


# --------------------------------------------------------------------------
# 月球（Meeus 低精度）
# --------------------------------------------------------------------------

def moon_position(when: dt.datetime):
    """月球地心视赤经/赤纬（度）与距离（km）。误差约 0.3°。"""
    jd = julian_day(when)
    t = (jd - 2451545.0) / 36525.0

    # 平黄经与主要幅角（度）
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t + t ** 3 / 538841.0
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t + t ** 3 / 545868.0
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t * t
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t + t ** 3 / 69699.0
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t - t ** 3 / 3526000.0

    dr, mr, mpr, fr = map(math.radians, (d, m, mp, f))

    # 黄经主要项（度）
    lon = lp + 6.289 * _SIN(mpr) + 1.274 * _SIN(2 * dr - mpr) \
        + 0.658 * _SIN(2 * dr) + 0.214 * _SIN(2 * mpr) - 0.186 * _SIN(mr) \
        - 0.114 * _SIN(2 * fr)
    # 黄纬主要项（度）
    lat = 5.128 * _SIN(fr) + 0.281 * _SIN(mpr + fr) - 0.278 * _SIN(fr - mpr) \
        - 0.173 * _SIN(2 * dr - fr)
    # 距离（km），用于视差修正（本原型仅影响升落阈值的微调）
    dist = 385311.64 - 20905.0 * _COS(mpr) - 3699.0 * _COS(2 * dr - mpr) \
        - 2956.0 * _COS(2 * dr)

    lon_r, lat_r = math.radians(lon), math.radians(lat)
    # 平黄赤交角
    eps = math.radians(23.43929111 - 0.0130042 * t)
    sin_lat, cos_lat = _SIN(lat_r), _COS(lat_r)
    ra = math.degrees(math.atan2(_SIN(lon_r) * _COS(eps) - math.tan(lat_r) * _SIN(eps),
                                 _COS(lon_r))) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0,
                                              sin_lat * _COS(eps)
                                              + cos_lat * _SIN(eps) * _SIN(lon_r)))))
    return ra, dec, dist


def _moon_altitude(when: dt.datetime, lat: float, lon: float) -> float:
    """月球地平高度（度）。"""
    ra, dec, _ = moon_position(when)
    ha = math.radians((local_sidereal_deg(when, lon) - ra + 180.0) % 360.0 - 180.0)
    latr, decr = math.radians(lat), math.radians(dec)
    sin_h = _SIN(latr) * _SIN(decr) + _COS(latr) * _COS(decr) * _COS(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_h))))


def _moon_hour_angle(when: dt.datetime, lon: float) -> float:
    """月球时角（度），范围 (−180, 180]，0 表示中天。"""
    ra, _, _ = moon_position(when)
    return ((local_sidereal_deg(when, lon) - ra + 180.0) % 360.0) - 180.0


def _interp_crossing(t0, f0, t1, f1):
    """在 f 由正变负（或相反）的区间内线性插值零点。"""
    if f1 == f0:
        return t0
    return t0 + dt.timedelta(hours=(t1 - t0).total_seconds() / 3600.0 * (f0 / (f0 - f1)))


def moon_times(date_utc: dt.datetime, lat: float, lon: float):
    """返回 (月出, 月中天, 月落)；当天不发生则为 None。

    做法：以 10 min 为步长采样当地时间一整天的高度与时角，
    再对过零点做线性插值（非线性误差 < 1 min，远优于业务所需的 ±10 min）。
    """
    base = dt.datetime(date_utc.year, date_utc.month, date_utc.day, tzinfo=UTC)
    step = dt.timedelta(minutes=10)
    samples = []
    for k in range(145):  # 24h + 1 个点，保证覆盖边界
        t = base + step * k
        samples.append((t, _moon_altitude(t, lat, lon), _moon_hour_angle(t, lon)))

    transit = rise = sset = None
    for (t0, h0, a0), (t1, h1, a1) in zip(samples, samples[1:]):
        # 中天：时角穿 0（跨 ±180° 的跳变要归一到同一支）
        if a0 * a1 < 0 and abs(a0 - a1) < 180.0:
            transit = _interp_crossing(t0, a0, t1, a1)
        # 升起：高度由负转正
        if h0 < _MOON_H0 <= h1:
            rise = _interp_crossing(t0, h0 - _MOON_H0, t1, h1 - _MOON_H0)
        # 落下：高度由正转负
        if h0 >= _MOON_H0 > h1:
            sset = _interp_crossing(t0, h0 - _MOON_H0, t1, h1 - _MOON_H0)
    return rise, transit, sset


# --------------------------------------------------------------------------
# Solunar 窗口
# --------------------------------------------------------------------------

# 缓存：{(utc 日期, 经度整数): 窗口列表}，避免钓鱼模型逐小时重复采样
_SOLUNAR_CACHE = {}


def solunar_windows(when: dt.datetime, lat: float, lon: float):
    """生成当天及前后各一天的 Solunar 窗口。

    返回 ``[(start_utc, end_utc, kind, strength), ...]``：
      * major（强度 1.0）：月中天 ±1 h、月下中天 ±1 h
      * minor（强度 0.65）：月出 ±1 h、月落 ±1 h

    月中天用精确月球时角过零求解；月下中天取 +12h25m（半朔望日），
    比“月中天+12h”更贴近真实月球日。
    """
    key = (when.date(), round(lat, 1), int(round(lon)))   # 月出/月落与纬度相关，缓存须含 lat
    cached = _SOLUNAR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SOLUNAR_CACHE) > 256:                          # 简单上限，防止长时间运行膨胀
        _SOLUNAR_CACHE.clear()

    out = []
    one_hour = dt.timedelta(hours=1)
    for delta_day in (-1, 0, 1):
        day = when + dt.timedelta(days=delta_day)
        rise, transit, sset = moon_times(day, lat, lon)
        if transit is not None:
            out.append((transit - one_hour, transit + one_hour, "major", 1.0))
            anti = transit + dt.timedelta(hours=12, minutes=25)
            out.append((anti - one_hour, anti + one_hour, "major", 1.0))
        for t in (rise, sset):
            if t is not None:
                out.append((t - one_hour, t + one_hour, "minor", 0.65))

    _SOLUNAR_CACHE[key] = out
    return out


def solunar_strength(when: dt.datetime, windows) -> float:
    """某时刻的 Solunar 强度：命中窗口取最大强度，否则给基础值 0.30。"""
    s = 0.30
    for a, b, _kind, strength in windows:
        if a <= when <= b:
            s = max(s, strength)
    return s
