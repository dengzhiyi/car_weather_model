# -*- coding: utf-8 -*-
"""三任务决策模型：山洪危险等级、露营适宜度、钓鱼指数/窗口。

三个模型都以 ``WeatherSeries + 起始下标`` 为输入，输出 JSON 片段；
它们不访问网络、不读文件，便于单独测试与在车机上做影子计算。

模型结构一致：
  1) 逐时打分（0–1 或 0–100）；
  2) 汇总为等级/结论（含硬否决）；
  3) 给出可解释 drivers/reasons（面向 HMI 文案，不做二次加工）。
"""

from __future__ import annotations

import datetime as dt
import math

from .astro import solunar_strength, solunar_windows, sun_times
from .common import iso_local, norm_cdf, prefix_sums, rnd, window_sum
from .physics import (apparent_temperature, scs_curve_number, scs_runoff, utci,
                      utci_category, wind_at_height)

# 等级 → 文案
_FLOOD_LABELS = ("极低", "低", "中等", "较高", "高")


# --------------------------------------------------------------------------
# 山洪
# --------------------------------------------------------------------------

def flood_model(w, idx: int, slope_deg: float, thresholds, horizon: int = 48):
    """山洪危险等级（临界雨量法 + API + SCS-CN 产流 + 坡度 + 土壤湿度）。

    评分思路
    --------
    ``ratio`` 取 1/3/6/24 h 累积雨量与当地临界雨量的最大比值（线性可解释）；
    再乘以四个修正因子：
      * f_api    前期降雨指数（K=0.87 的指数衰减，土体已饱和则更易产流）
      * f_soil   表层土壤湿度（0–1 cm 体积含水量）
      * f_slope  坡度（tan 归一化，陡坡汇流快）
      * f_runoff SCS-CN 产流系数偏离基准的程度（替代纯阈值比的“非线性”修正）
    最后用对数正态 + erf 得到 5 级概率分布，``σ`` 随预报时效增大。
    """
    precip = w["precipitation"]
    times = w["times"]
    n = len(precip)
    ps = prefix_sums(precip)          # 所有窗口求和 O(1)

    # 前期降雨指数：往前 24 小时，权重 K^(k-1)
    api, k = 0.0, 0.87
    for back, j in enumerate(range(idx - 1, max(-1, idx - 25), -1), start=1):
        api += (k ** (back - 1)) * precip[j]
    f_api = 1.0 + min(0.35, api / 120.0)

    f_slope = 0.9 + 0.4 * min(1.0, math.tan(math.radians(max(0.0, slope_deg))) / 0.7)

    best = {"s": -1.0}
    onset = None
    peak_level = 1
    for i in range(idx, min(n, idx + horizon)):
        r1 = window_sum(ps, i, 1)
        r3 = window_sum(ps, i, 3)
        r6 = window_sum(ps, i, 6)
        r24 = window_sum(ps, i, 24)
        ratio = max(r1 / thresholds["1h"], r3 / thresholds["3h"],
                    r6 / thresholds["6h"], r24 / thresholds["24h"])

        sm = w["soil_moisture"][i]
        f_soil = 1.2 if sm >= 0.36 else (0.85 if sm <= 0.15 else 1.0)

        # SCS-CN：由当前土壤湿度与坡度推 CN，再算 24h 累积产流
        cn = scs_curve_number(sm, slope_deg)
        runoff = scs_runoff(r24, cn)
        runoff_coef = runoff / r24 if r24 > 0.5 else 0.0
        # 基准产流系数 0.35；偏离越大越危险，系数限制在 [0.9, 1.3] 避免压过主项
        f_runoff = max(0.9, min(1.3, 1.0 + 0.6 * (runoff_coef - 0.35)))

        s = ratio * f_soil * f_slope * f_api * f_runoff
        lvl = 1 if s < 0.45 else 2 if s < 0.75 else 3 if s < 1.0 else 4 if s < 1.35 else 5
        if lvl >= 3 and onset is None:
            onset = times[i]
        peak_level = max(peak_level, lvl)
        if s > best["s"]:
            best = {"s": s, "i": i, "r1": r1, "r3": r3, "r6": r6, "r24": r24, "sm": sm,
                    "level": lvl, "ratio": ratio, "cn": cn, "runoff": runoff}

    # 概率分布：以峰值评分 s_peak 为均值（对数域），σ 随提前时效增长
    s_peak = max(best["s"], 1e-6)
    lead_h = max(0.0, (times[best["i"]] - times[idx]).total_seconds() / 3600.0)
    sig = 0.12 + 0.018 * lead_h

    def p_ge(level_score):
        return norm_cdf((math.log(s_peak) - math.log(level_score)) / sig)

    p2, p3, p4, p5 = p_ge(0.45), p_ge(0.75), p_ge(1.0), p_ge(1.35)
    dist = [max(0.0, 1 - p2), max(0.0, p2 - p3), max(0.0, p3 - p4),
            max(0.0, p4 - p5), max(0.0, p5)]
    total = sum(dist) or 1.0
    dist = [x / total for x in dist]

    drivers = []
    if best["r3"] >= thresholds["3h"]:
        drivers.append(f"3h雨量 {best['r3']:.1f}mm ≥ 临界 {thresholds['3h']:.0f}mm")
    if best["r1"] >= thresholds["1h"]:
        drivers.append(f"1h雨量 {best['r1']:.1f}mm ≥ 临界 {thresholds['1h']:.0f}mm")
    drivers.append(f"坡度 {slope_deg:.0f}°")
    drivers.append(f"土壤湿度 {best['sm']:.2f} m³/m³")
    if best["runoff"] > 0.5:
        drivers.append(f"SCS-CN 产流 {best['runoff']:.1f}mm（CN={best['cn']:.0f}）")
    if api > 20:
        drivers.append(f"前期降雨指数 API={api:.0f}")

    return {
        "level": peak_level,
        "label": _FLOOD_LABELS[peak_level - 1],
        "prob": [rnd(x, 3) for x in dist],
        "peak_time": iso_local(times[best["i"]]),
        "onset_time": iso_local(onset) if onset else None,
        "score_ratio": rnd(best["ratio"], 2),
        "runoff_24h_mm": rnd(best["runoff"], 1),
        "curve_number": rnd(best["cn"], 0),
        "drivers": drivers,
        "thresholds": {k: rnd(v, 1) for k, v in thresholds.items()
                       if k in ("1h", "3h", "6h", "24h")},
        "threshold_source": thresholds.get("source"),
    }


# --------------------------------------------------------------------------
# 露营
# --------------------------------------------------------------------------

def comfort_index(t_c: float, rh: float, wind10: float, radiation: float, cloud: float):
    """体感温度：优先精确 UTCI，异常时退化为 Steadman AT。

    平均辐射温度 Tmrt 的估计（只有短波辐照可用，故采用经验近似）：
      * 有日照：Tmrt ≈ Ta + 0.02·G（G 为总水平辐照，W/m²）
      * 无日照：Tmrt ≈ Ta − 2·(1 − 云量)，晴夜辐射冷却更强
    风速：Open-Meteo 为 10 m 风速，UTCI 官方要求也是 10 m，直接使用；
    降级路径的 AT 需要人体高度风速，故用对数风廓线降到 1.1 m。
    """
    wind10 = max(0.0, wind10)
    tmrt = t_c + 0.02 * radiation if radiation > 0 else t_c - 2.0 * (1.0 - cloud / 100.0)
    try:
        value = utci(t_c, tmrt, wind10, rh)
        return value, "utci", utci_category(value)
    except Exception:
        return (apparent_temperature(t_c, rh, wind_at_height(wind10, 1.1)),
                "apparent_temperature", "n/a")


def camping_model(w, idx: int, stay_hours: float = 12):
    """露营适宜度（0–100）与连续适宜窗口。

    五因子加权：体感 0.42 / 降水 0.22 / 风 0.16 / 露点差 0.12 / UV 0.08。
    硬否决（阵风≥7级、暴雨、雷暴、体感极端）一旦出现，总分封顶 45，
    避免出现“81 分但不建议”的自相矛盾结论。
    """
    hours = []
    vetoes = []
    end = min(len(w["times"]), idx + int(stay_hours))

    for i in range(idx, end):
        t = w["temperature"][i]
        rh = w["humidity"][i]
        td = w["dewpoint"][i]
        wind = w["wind"][i]
        gust = w["gust"][i]
        rad = w["radiation"][i]
        cloud = w["cloud"][i]
        precip = w["precipitation"][i]
        prob = w["precip_prob"][i] / 100.0 if w["precip_prob"][i] > 1 else w["precip_prob"][i]
        uv = w["uv"][i]
        cape = w["cape"][i]

        eff, method, category = comfort_index(t, rh, wind, rad, cloud)
        s_comfort = math.exp(-((eff - 22.0) / 11.0) ** 2)
        s_rain = max(0.0, 1.0 - max(precip / 8.0, prob / 0.9))
        s_wind = math.exp(-((wind - 3.5) / 4.5) ** 2)
        dd = t - td
        s_dew = 1.0 if dd >= 4 else (0.45 if dd >= 2 else 0.15)
        s_uv = 1.0 if uv <= 7 else max(0.2, 1.0 - (uv - 7) / 6.0)
        score = (0.42 * s_comfort + 0.22 * s_rain + 0.16 * s_wind
                 + 0.12 * s_dew + 0.08 * s_uv) * 100.0

        why = []
        if gust >= 15:
            why.append("阵风≥7级")
        if precip >= 8 or (prob >= 0.85 and precip >= 4):
            why.append("暴雨")
        if cape >= 1000 and prob >= 0.6:
            why.append("雷暴风险")
        if eff <= -13 or eff >= 38:
            why.append("体感极端")
        if why:
            vetoes.append({"time": iso_local(w["times"][i]), "why": why})

        hours.append({
            "time": iso_local(w["times"][i]), "score": score, "t": t, "rh": rh,
            "dew_gap": dd, "wind": wind, "gust": gust, "precip": precip,
            "prob": prob, "eff": eff, "method": method, "category": category,
            "cloud": cloud,
        })

    if not hours:
        raise ValueError("stay window 超出预报范围")

    avg = sum(h["score"] for h in hours) / len(hours)
    if vetoes:
        avg = min(avg, 45.0)
    worst = min(hours, key=lambda h: h["score"])
    verdict = "不建议" if vetoes else ("适宜" if avg >= 65 else "需谨慎" if avg >= 45 else "不建议")

    reasons = []
    if vetoes:
        reasons.append(f"{vetoes[0]['time']} {'/'.join(vetoes[0]['why'])}")
    reasons.append(f"体感 {worst['eff']:.0f}（{worst['method']}）")
    if worst["dew_gap"] < 2:
        reasons.append(f"最湿时段露点差 {worst['dew_gap']:.1f}℃ 易结露")
    max_wind = max(h["wind"] for h in hours)
    if max_wind >= 6:
        reasons.append(f"最大风速 {max_wind:.1f} m/s")
    max_prob = max(h["prob"] for h in hours)
    if max_prob >= 0.5:
        reasons.append(f"最高降水概率 {max_prob * 100:.0f}%")
    reasons.append(f"夜间最低温 {min(h['t'] for h in hours):.1f}℃")

    # 连续适宜窗口：score ≥ 60 且无否决
    windows, cur = [], None
    for h in hours:
        if h["score"] >= 60 and not vetoes:
            if cur is None:
                cur = [h["time"], h["time"], h["score"], h["score"]]
            else:
                cur[1] = h["time"]
                cur[2] = max(cur[2], h["score"])
                cur[3] = min(cur[3], h["score"])
        elif cur is not None:
            windows.append(cur)
            cur = None
    if cur is not None:
        windows.append(cur)

    return {
        "score": rnd(avg, 0),
        "verdict": verdict,
        "veto": vetoes[0]["why"] if vetoes else None,
        "window": f"{windows[0][0]} ~ {windows[0][1]}" if windows else None,
        "reasons": reasons[:4],
        "min_temp": rnd(min(h["t"] for h in hours), 1),
        "max_gust": rnd(max(h["gust"] for h in hours), 1),
        "condensation_hours": sum(1 for h in hours if h["dew_gap"] < 2),
    }


# --------------------------------------------------------------------------
# 钓鱼
# --------------------------------------------------------------------------

def _water_temp(w, idx: int, horizon: int):
    """水温滞后模型：以过去 72 h 气温均值为初值，再按 α=0.03/h 一阶滞后跟踪气温。

    水体热惯性远大于空气，这也是“降温当天鱼口差、次日才转好”的物理来源。
    """
    temps = w["temperature"]
    lo = max(0, idx - 72)
    window = temps[lo:idx + 1]
    tw = sum(window) / max(1, len(window))
    out = []
    for i in range(idx, min(len(temps), idx + horizon)):
        tw += 0.03 * (temps[i] - tw)
        out.append(tw)
    return out


def fishing_model(w, idx: int, lat: float, lon: float, horizon: int = 48):
    """钓鱼指数（0–100）与可钓窗口。

    评分 = 天气项 0.40 + Solunar 项 0.55 + 晨昏低光加成 0.05。
    Solunar 用精确月历（月中天 / 月下中天 / 月出 / 月落 ±1 h），
    天气项包含气压 24 h 趋势、水温、风速、云量、降水。

    窗口条件：连续 ≥2 h 且逐时 score ≥ 65、无强降水/大风否决。
    """
    times = w["times"]
    pmsl = w["pressure_msl"]
    tw = _water_temp(w, idx, horizon)
    start = times[idx]

    # 日出日落优先取本地天文计算（不依赖 API 是否返回 daily 字段）
    sr, ss = sun_times(start, lat, lon)
    if sr is None:
        sr = min([s for s in w["sunrise"] if s >= start] or [start + dt.timedelta(hours=6)])
        ss = min([s for s in w["sunset"] if s >= start] or [start + dt.timedelta(hours=18)])
    elif ss is not None and ss < start:
        # 出发时刻已过当天日落：改取“下一个”日出日落，夜间时段才能正确判为低光
        sr, ss = sun_times(start + dt.timedelta(days=1), lat, lon)
    sunrise0, sunset0 = sr, ss

    sol_windows = solunar_windows(start, lat, lon)

    rows = []
    for k, i in enumerate(range(idx, min(len(times), idx + horizon))):
        t = times[i]
        sol = solunar_strength(t, sol_windows)
        dp24 = pmsl[i] - pmsl[max(0, i - 24)]
        p_score = max(0.15, min(1.0, 0.5 - dp24 / 8.0))
        wt = tw[k]
        w_score = math.exp(-((wt - 20.0) / 8.0) ** 2)
        wind = w["wind"][i]
        wind_score = math.exp(-((wind - 4.0) / 3.2) ** 2)
        cloud = w["cloud"][i] / 100.0
        cloud_score = 1.0 - 0.7 * abs(cloud - 0.6)
        precip = w["precipitation"][i]
        rain_score = 1.0 if precip < 0.5 else (0.55 if precip < 2 else 0.3)

        # 低光加成：夜间、日出后 2 h 内、日落前 2 h 内
        night = t < sunrise0 or t > sunset0
        lowlight = (night or t < sunrise0 + dt.timedelta(hours=2)
                    or t >= sunset0 - dt.timedelta(hours=2))

        weather = (0.25 * p_score + 0.15 * w_score + 0.15 * wind_score
                   + 0.10 * cloud_score + 0.05 * rain_score) / 0.70
        score = (0.40 * weather + 0.55 * sol + (0.05 if lowlight else 0.0)) * 100.0
        veto = precip >= 6.0 or w["gust"][i] >= 17.0
        rows.append({"i": i, "score": min(100.0, score), "veto": veto, "sol": sol,
                     "dp24": dp24, "wt": wt, "wind": wind, "precip": precip})

    # 连续窗口
    windows, cur = [], None
    for r in rows:
        if r["score"] >= 65 and not r["veto"]:
            if cur is None:
                cur = {"start": times[r["i"]], "end": times[r["i"]], "peak": r["score"],
                       "hours": 1}
            else:
                cur["end"] = times[r["i"]]
                cur["peak"] = max(cur["peak"], r["score"])
                cur["hours"] += 1
        elif cur is not None:
            windows.append(cur)
            cur = None
    if cur is not None:
        windows.append(cur)
    windows = [x for x in windows if x["hours"] >= 2]
    windows.sort(key=lambda x: -x["peak"])
    windows = windows[:4]
    windows.sort(key=lambda x: x["start"])

    best = max(rows, key=lambda r: r["score"])
    drivers = []
    if best["sol"] >= 0.9:
        drivers.append("命中 Solunar 主期")
    elif best["sol"] >= 0.6:
        drivers.append("命中 Solunar 次期")
    drivers.append(f"气压24h变化 {best['dp24']:+.1f}hPa")
    drivers.append(f"水温估计 {best['wt']:.1f}℃")
    drivers.append(f"风速 {best['wind']:.1f} m/s")
    drivers.append(f"降水 {best['precip']:.1f} mm/h")

    return {
        "score": rnd(best["score"], 0),
        "windows": [f"{iso_local(x['start'])} ~ {iso_local(x['end'])}" for x in windows],
        "best_time": iso_local(times[best["i"]]),
        "drivers": drivers,
        "solunar_note": "Meeus 月历 + GMST，窗口中天/月出月落 ±1h",
    }


# --------------------------------------------------------------------------
# 结论与置信度
# --------------------------------------------------------------------------

def confidence(lead_hours: float, source: str, stage: str) -> float:
    """置信度：时效越长越低；合成数据 −0.15；抵达校正后 +0.25 并封顶 0.93。"""
    c = 0.9 - 0.015 * lead_hours
    if source != "open-meteo":
        c -= 0.15
    if stage == "post_arrival":
        c = min(0.93, c + 0.25)
    return max(0.15, min(0.95, c))


def actions(flood, camp, fish):
    """把三个模型结论压缩成 3–4 条 HMI 行动建议（按风险优先级排序）。"""
    out = []
    if flood["level"] >= 3:
        out.append("山洪风险中等以上：不要在沟谷、河滩、干涸河床停留，留意上游降雨")
    if flood["level"] >= 4:
        out.append("建议改期或撤离至高处空旷地")
    if camp["verdict"] != "适宜":
        detail = ("不建议：" + str(camp["veto"])) if camp["veto"] else "需谨慎，加固帐篷并备防雨"
        out.append(f"露营{detail}")
    if camp.get("condensation_hours", 0) >= 2:
        out.append("易结露：使用地布、睡袋防潮")
    if fish["windows"]:
        out.append(f"钓鱼可用窗口：{fish['windows'][0]}")
    out.append("出发前 3 小时复核一次（山洪时效敏感）")
    return out[:4]
