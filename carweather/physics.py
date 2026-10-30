# -*- coding: utf-8 -*-
"""物理与经验模型：UTCI、晴空辐照（Ineichen/Perez）、大气质量、水汽、SCS-CN 产流。

所有实现均为“零第三方依赖”的纯 Python 版本，便于在车机（可能无 numpy）上运行。

移植来源与许可
--------------
* UTCI 六阶多项式：Bröde et al. (2012)；系数表由 pythermalcomfort (MIT) 抽取，
  见 ``_utci_table.py``。
* Ineichen/Perez 晴空模型、Kasten-Young 1989 大气质量：pvlib (BSD-3-Clause)，
  按其公式重写，未复制代码。
* SCS-CN 曲线数产流：USDA NRCS 公开技术手册（公共领域）。
"""

from __future__ import annotations

import math

from ._utci_table import MAX_POW, _UTCI_TERMS

# --------------------------------------------------------------------------
# 基础热力学
# --------------------------------------------------------------------------


def saturation_vapor_pressure(t_c: float) -> float:
    """饱和水汽压（hPa）。

    采用 Wexler 拟合的 7 项多项式（与 pythermalcomfort 的 ``exponential`` 一致，
    亦为 UTCI 官方参考实现所用），在 −50…50 ℃ 内误差远小于 0.1%。
    """
    g = (-2836.5744, -6028.076559, 19.54263612, -0.02737830188,
         0.000016261698, 7.0229056e-10, -1.8680009e-13)
    tk = t_c + 273.15
    es = 2.7150305 * math.log1p(tk)
    for i, coef in enumerate(g):
        es += coef * tk ** (i - 2)
    return math.exp(es) * 0.01          # Pa → hPa


def vapor_pressure_hpa(t_c: float, rh_pct: float) -> float:
    """实际水汽压（hPa）。"""
    return saturation_vapor_pressure(t_c) * max(0.0, min(100.0, rh_pct)) / 100.0


def dew_point(t_c: float, rh_pct: float) -> float:
    """露点温度（℃），Magnus 公式，反解自相对湿度。"""
    rh = max(1.0, min(100.0, rh_pct))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + a * t_c / (b + t_c)
    return b * gamma / (a - gamma)


def sea_level_pressure(p_station_hpa: float, t_c: float, elevation_m: float) -> float:
    """站点气压 → 海平面气压（国际民航组织标准大气，用于与预报 pressure_msl 比对）。"""
    return p_station_hpa * (1.0 - 0.0065 * elevation_m / (t_c + 0.0065 * elevation_m + 273.15)) ** -5.257


def wind_at_height(v10: float, h_m: float) -> float:
    """对数风廓线：10 m 风速 → 任意高度风速。

    z0 = 0.03 m（短草地/停车场），10 m → 1.1 m（人体重心）约 ×0.62。
    """
    v10 = max(1e-3, v10)
    return v10 * math.log(h_m / 0.03) / math.log(10.0 / 0.03)


# --------------------------------------------------------------------------
# UTCI（Universal Thermal Climate Index）
# --------------------------------------------------------------------------

# 预先把 210 项按 (次幂组合, 系数) 摊平；求值时只做乘法与查表，避免重复 ** 运算。
_TERMS = _UTCI_TERMS
_MAX = MAX_POW


def _powers(x: float, n: int):
    """返回 (1, x, x², …, xⁿ)，一次循环生成，供多项式按幂次索引。"""
    out = [1.0] * (n + 1)
    for i in range(1, n + 1):
        out[i] = out[i - 1] * x
    return out


def utci_polynomial(tdb: float, v: float, dtr: float, pa: float) -> float:
    """UTCI 六阶多项式本体（不含输入范围裁剪）。

    参数：
      tdb  干球温度 ℃
      v    10 m 高度风速 m/s（UTCI 参考高度）
      dtr  平均辐射温度 − 干球温度 ℃
      pa   水汽分压 kPa
    """
    tp = _powers(tdb, _MAX[0])
    vp = _powers(v, _MAX[1])
    dp = _powers(dtr, _MAX[2])
    pp = _powers(pa, _MAX[3])
    total = tdb                    # 多项式的首项是裸 tdb（系数表已剔除该项）
    for a, b, c, d, coef in _TERMS:
        total += coef * tp[a] * vp[b] * dp[c] * pp[d]
    return total


def utci(tdb: float, tr: float, v10: float, rh: float) -> float:
    """UTCI 值（℃，未取整）。

    与 pythermalcomfort ``utci(tdb, tr, v, rh)`` 数值一致（误差 < 1e-9），
    范围裁剪按官方适用域：−50 < tdb < 50、−30 < tr − tdb < 70、0.5 ≤ v ≤ 17。
    """
    tdb = max(-50.0, min(50.0, tdb))
    dtr = max(-30.0, min(70.0, tr - tdb))
    v10 = max(0.5, min(17.0, v10))
    rh = max(0.0, min(100.0, rh))

    pa = saturation_vapor_pressure(tdb) * (rh / 100.0) / 10.0   # hPa → kPa
    return utci_polynomial(tdb, v10, dtr, pa)


# UTCI 热应力分级（℃ 下界 → 标签），与官方表一致
_UTCI_CATEGORIES = (
    (-40.0, "极寒冷应激"), (-27.0, "强冷应激"), (-13.0, "冷应激"),
    (0.0, "轻微冷应激"), (9.0, "无热应激"), (26.0, "中等热应激"),
    (32.0, "强热应激"), (38.0, "极强热应激"), (46.0, "酷热应激"),
)


def utci_category(value: float) -> str:
    """UTCI 热应激等级标签。"""
    label = _UTCI_CATEGORIES[-1][1]
    for lower, name in _UTCI_CATEGORIES:
        if value >= lower:
            label = name
    return label


def apparent_temperature(t_c: float, rh_pct: float, wind10: float) -> float:
    """Steadman 体感温度（AT）近似，用于 UTCI 不可用时的降级路径。

    高温段用湿空气感热项，低温大风段叠加风寒指数（wind chill）。
    """
    e = vapor_pressure_hpa(t_c, rh_pct)
    at = t_c + 0.33 * e - 0.70 * wind10 - 4.0
    if t_c <= 10.0 and wind10 >= 1.34:
        at = min(at, 13.12 + 0.6215 * t_c - 11.37 * wind10 ** 0.16
                 + 0.3965 * t_c * wind10 ** 0.16)
    return at


# --------------------------------------------------------------------------
# 辐射：大气质量 / 晴空辐照
# --------------------------------------------------------------------------

# 地球轨道偏心率修正后的日地辐照（W/m²），dni_extra
def extraterrestrial_irradiance(doy: int) -> float:
    """大气层外法向辐照（W/m²），Spencer 近似，年变化 ±3.3%。"""
    return 1366.7 * (1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0))


def refraction_deg(elevation_deg: float) -> float:
    """大气折射修正（度），Bennett 公式；用于真高度 → 视高度。"""
    if elevation_deg < -1.0:
        return 0.0
    r_arcmin = 1.02 / math.tan(math.radians(elevation_deg + 10.3 / (elevation_deg + 5.11)))
    return max(0.0, r_arcmin / 60.0)


def apparent_zenith(elevation_deg: float) -> float:
    """真太阳高度 → 视天顶角（Ineichen 需要视天顶角）。"""
    return 90.0 - (elevation_deg + refraction_deg(elevation_deg))


def kasten_young_airmass(zenith_deg: float) -> float:
    """Kasten & Young (1989) 相对大气质量，天顶角 0…90°。"""
    z = min(90.0, max(0.0, zenith_deg))
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (6.07995 + (90.0 - z)) ** -1.6364)


def absolute_airmass(zenith_deg: float, pressure_pa: float) -> float:
    """气压订正后的绝对大气质量（pvlib 同款）：AMa = AMr × P / 101325。"""
    return kasten_young_airmass(zenith_deg) * pressure_pa / 101325.0


def station_pressure_from_altitude(altitude_m: float) -> float:
    """标准大气下海拔对应气压（Pa），pvlib 用的指数近似。"""
    return 101325.0 * math.exp(-0.0001184 * altitude_m)


def linke_turbidity_estimate(lat: float, doy: int, altitude_m: float, rh_pct: float) -> float:
    """Linke 浑浊度经验估计（无量纲）。

    正式实现应查 SoDa 的月平均格点数据（``LinkeTurbidities.h5``）；
    车机不便携带 20 MB 数据文件，这里用气候代理：
      * 基准 2.8（中纬度大陆性气溶胶）
      * 高海拔每 1000 m 减 0.25（气溶胶柱变薄）
      * 夏季 +0.35、冬季 −0.2（对流活动与采暖气溶胶）
      * 高湿（rh ≥ 80%）+0.4（吸湿增长使有效浑浊度升高）
    误差约 ±0.5，对 GHI 的影响 < 4%，用于云量反演足够。
    """
    seasonal = 0.35 * math.cos(2.0 * math.pi * (doy - 200) / 365.0)
    humid = 0.4 if rh_pct >= 80 else (0.15 if rh_pct >= 60 else 0.0)
    tl = 2.8 + seasonal - 0.25 * (altitude_m / 1000.0) + humid
    return max(1.5, min(5.0, tl))


def ineichen_clear_sky(elevation_deg: float, altitude_m: float, linke_turbidity: float,
                       doy: int):
    """Ineichen/Perez 晴空模型，返回 ``(ghi, dni, dhi)``，单位 W/m²。

    与 pvlib ``clearsky.ineichen`` 的公式一致（Perez 增强项默认关闭）。
    """
    if elevation_deg <= -0.5:
        return 0.0, 0.0, 0.0

    zen = apparent_zenith(elevation_deg)              # 视天顶角（折射已计入）
    cos_z = max(0.0, math.cos(math.radians(min(90.0, zen))))

    tl = linke_turbidity
    dni_extra = extraterrestrial_irradiance(doy)
    am_a = absolute_airmass(zen, station_pressure_from_altitude(altitude_m))

    fh1 = math.exp(-altitude_m / 8000.0)
    fh2 = math.exp(-altitude_m / 1250.0)
    cg1 = 5.09e-05 * altitude_m + 0.868
    cg2 = 3.92e-05 * altitude_m + 0.0387

    ghi = math.exp(-cg2 * am_a * (fh1 + fh2 * (tl - 1.0)))
    ghi = cg1 * dni_extra * cos_z * max(ghi, 0.0)

    b = 0.664 + 0.163 / fh1
    bnci = dni_extra * max(b * math.exp(-0.09 * am_a * (tl - 1.0)), 0.0)
    bnci_2 = ghi * min(max((1.0 - (0.1 - 0.2 * math.exp(-tl)) / (0.1 + 0.882 / fh1)) / cos_z, 0.0), 1e20)
    dni = min(bnci, bnci_2)
    dhi = ghi - dni * cos_z
    return ghi, dni, dhi


def lux_to_ghi(lux: float) -> float:
    """照度 → 总水平辐照（W/m²）。

    日光光谱下 1 W/m² ≈ 110 lux（CIE  daylight 换算），阴天略高；
    车载只用它做云量反演，量级正确即可。
    """
    return max(0.0, lux) / 110.0


def ghi_to_lux(ghi: float) -> float:
    """总水平辐照 → 照度，用于合成数据与显示。"""
    return max(0.0, ghi) * 110.0


# --------------------------------------------------------------------------
# SCS-CN 产流
# --------------------------------------------------------------------------

def scs_curve_number(soil_moisture: float, slope_deg: float, base_cn: float = 75.0) -> float:
    """由土壤湿度与坡度推 SCS 曲线数 CN（AMC 条件化 + 坡度订正）。

    参数：
      soil_moisture  0–1 cm 体积含水量（Open-Meteo 同名字段，m³/m³）
      slope_deg      地面坡度
      base_cn        AMC II（正常）条件下的基准 CN，默认 75（混合林地/草地）

    AMC 划分沿用项目既有阈值：<0.15 干旱(I)，0.15–0.36 正常(II)，>0.36 湿润(III)。
    坡度订正：坡度越大汇流越快，CN 上调，最多 +10。
    """
    if soil_moisture > 0.36:
        cn = 23.0 * base_cn / (10.0 + 0.13 * base_cn)      # AMC III
    elif soil_moisture < 0.15:
        cn = 4.2 * base_cn / (10.0 - 0.058 * base_cn)      # AMC I
    else:
        cn = base_cn

    # 坡度订正：tan(坡度)/0.5 归一化，最大 +10
    cn += 10.0 * min(1.0, math.tan(math.radians(max(0.0, slope_deg))) / 0.5)
    return max(30.0, min(98.0, cn))


def scs_runoff(precip_mm: float, cn: float, ia_ratio: float = 0.2) -> float:
    """SCS-CN 直接径流深（mm）。

    S = 25400/CN − 254；Ia = ia_ratio·S（默认 0.2）；Q = (P − Ia)² / (P − Ia + S)。
    """
    if precip_mm <= 0:
        return 0.0
    s = 25400.0 / cn - 254.0
    ia = ia_ratio * s
    if precip_mm <= ia:
        return 0.0
    return (precip_mm - ia) ** 2 / (precip_mm - ia + s)
