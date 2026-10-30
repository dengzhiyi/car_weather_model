# carweather — 车载气象传感器 + Open-Meteo 的三任务决策原型

用**车载 5 个基础传感器**（温度、湿度、气压、雨量、光线）叠加 **Open-Meteo 预报/再分析**，
输出三类与生活/安全直接相关的结果：

1. **山洪危险等级**（1–5 级 + 5 级概率分布）
2. **露营建议**（0–100 分 + 结论 + 连续适宜窗口 + 硬否决）
3. **钓鱼指数 / 钓鱼窗口**（0–100 分 + Solunar 窗口 + 最佳时段）

设计目标是**车机可离线运行、结果可解释、算法可追溯**：运行时零第三方依赖（纯标准库），
网络不可用时自动降级为确定性合成数据，并在输出中标注数据来源。

---

## 1. 硬约束

| 约束 | 实现方式 |
|---|---|
| 传感器只有 5 个（温度/湿度/气压/雨量/光线） | 不假设存在风速、辐射、土壤湿度等本地量，全部由预报或模型推导 |
| 本地传感器**仅在驻车状态**参与计算 | `mode=parked` 或 `speed < 3 km/h` 才采样；行驶数据一律丢弃 |
| 驻车后需要热平衡 | 丢弃驻车开始后前 **5 分钟**预热数据 |
| **出发前给建议，到达后实测校正** | 两段式 `predict` → `correct`，输出结构同构，仅 `stage` 不同 |
| 车机可能无网络 | 一次预测仅 3 个 HTTP 请求；任何失败自动降级 `synthetic` |
| 结果必须可解释 | 每个模型输出 `drivers` / `reasons` / `veto`，不做二次文案加工 |

---

## 2. 总体架构

```
                 ┌──────────────────────────────┐
                 │  Open-Meteo                  │
                 │  forecast / archive(ERA5)    │
                 │  elevation (9 点批量)        │
                 └──────────────┬───────────────┘
                                │ sources.py（缓存 + 降级）
                                ▼
   ┌──────────────┐      ┌───────────────┐      ┌────────────────┐
   │ 车载 5 传感器 │─────▶│  WeatherSeries │─────▶│  models.py     │
   │ 驻车 CSV     │ 同化  │  + 观测快照 obs │      │  flood/camp/   │
   └──────────────┘      └───────────────┘      │  fishing       │
          │                     ▲                └───────┬────────┘
          │                     │                        │
          │              pipeline.py                     │
          │        predict / correct（两段式）            │
          ▼                     │                        ▼
   aggregate_obs                │                 actions + meta
   （QC / 预热 / 云量反演）       └──────────────▶  JSON（stdout / --out）
```

### 两段式流程

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| `pre_departure`（出发前） | 目的地坐标 + ETA + 停留时长 | 纯预报推理；山洪用 ERA5 历史分位定本地临界雨量 | `meta/flood/camping/fishing/actions` |
| `post_arrival`（抵达驻车） | 驻车传感器 CSV + 坐标（ETA 可选） | 观测 QC → 单点最优插值同化 → 三模型重算 | 同上 + `observed / corrections / delta / baseline` |

同化采用单点最优插值（OI）：

```
X = X_fcst + w · (X_obs − X_fcst)
w = clamp(0.75 − 0.12 · len(qc_flags), 0.4, 0.75)
```

- 只修正 ETA 所在小时这一列，后续小时仍保留预报的演变趋势；
- 降水取 `max(预报, 0.6 × 实测 1h 雨量)`，避免把正在下的雨“抹平”；
- 光线反演云量：`kt = G_obs / G_clear`，`cloud = clamp(1 − kt, 0, 1)`；
  太阳高度 < 10° 或实测 < 25% 晴空值时不做反演（判为隧道/树荫/夜间）。

---

## 3. 代码结构

```
/workspace
├── car_weather.py              # 15 行兼容入口：from carweather.cli import main
├── carweather/
│   ├── __init__.py           # 版本与高层导出（predict / correct / make_sample）
│   ├── common.py             # 时区、修约、前缀和窗口、分位/正态 CDF、最近邻索引
│   ├── astro.py              # NOAA 太阳位置/升落、Meeus 月球、GMST/LST、Solunar
│   ├── physics.py            # UTCI、Ineichen 晴空、Kasten-Young、水汽、SCS-CN
│   ├── _utci_table.py        # 210 项 UTCI 系数表（自动抽取 + 校验说明）
│   ├── sources.py            # Open-Meteo / ERA5 / 高程 / 传感器 CSV / 合成数据
│   ├── models.py             # 山洪、露营、钓鱼三模型 + 置信度 + 行动建议
│   ├── pipeline.py           # 两段式主流程、观测聚合、同化
│   └── cli.py                # argparse；stdout 只输出 JSON，人类可读文本走 stderr
└── samples/
    └── sensor_parked.csv     # 100 分钟驻车样例（含噪声与 +0.6℃ 系统偏差）
```

分层原则：**数据接入 → 物理/天文基础 → 业务模型 → 流程编排 → CLI**，
下层不依赖上层；`models.py` 不访问网络、不读文件，可单独测试。

---

## 4. 算法与移植

### 4.1 已移植（可直接使用）

| 能力 | 实现位置 | 来源 / 许可 | 说明 |
|---|---|---|---|
| **UTCI 精确模型** | [physics.py](carweather/physics.py) + `_utci_table.py` | Bröde et al. (2012)；系数抽取自 pythermalcomfort（MIT） | 210 项六阶多项式 + Wexler 7 项水汽压；与参考实现 3000 组随机输入最大偏差 **8.7e-12** |
| **Ineichen/Perez 晴空辐照** | [physics.py](carweather/physics.py) | pvlib（BSD-3-Clause），按公式重写 | 需要视天顶角、绝对大气质量、Linke 浑浊度；返回 GHI/DNI/DHI |
| **Kasten-Young 1989 大气质量** | [physics.py](carweather/physics.py) | pvlib（BSD-3-Clause） | 天顶角 90° 时 AM=37.92（与 pvlib 一致） |
| **气压订正大气质量** | [physics.py](carweather/physics.py) | pvlib | `AMa = AMr × P / 101325` |
| **SCS-CN 产流** | [physics.py](carweather/physics.py) | USDA NRCS 公开手册（公共领域） | AMC I/II/III + 坡度订正；`S=25400/CN−254`，`Q=(P−0.2S)²/(P−0.2S+S)` |
| **太阳位置与升落** | [astro.py](carweather/astro.py) | NOAA 低精度公式 | 赤纬 + 均时差；升落阈值 −0.833°；与 astral 对比 **±1.6 min** |
| **月球位置与升落/中天** | [astro.py](carweather/astro.py) | Meeus《Astronomical Algorithms》第 47 章 | 主要周期项，赤经/赤纬误差约 0.3°；升落阈值 +0.125°（折射−视差） |
| **Solunar 窗口** | [astro.py](carweather/astro.py) | 自研（思想参考 SolunarPeriods，GPL-3.0，仅参考未抄码） | 月中天=时角过零；下中天=+12h25m；月出/月落 ±1h；按 `(日期, 纬度, 经度)` 缓存 |

### 4.2 关键实现细节

**UTCI**：系数表在导入时展开为 `(tdb幂, v幂, dtr幂, pa幂, 系数)`；
求值只做幂表查值与乘法，单次 30 µs。输入按官方适用域裁剪：
`−50 < tdb < 50`、`−30 < tr−tdb < 70`、`0.5 ≤ v ≤ 17`。
平均辐射温度因只有短波辐照可用，采用经验近似：
有日照 `Tmrt ≈ Ta + 0.02·G`，无日照 `Tmrt ≈ Ta − 2(1 − 云量)`。
（注释与代码中均标明这是近似，不是辐射计实测。）

**Linke 浑浊度**：正式做法应查 SoDa 月平均格点数据（约 20 MB），车机不携带，
改用气候代理：基准 2.8 + 季节项 ±0.35 − 海拔 0.25/1000m + 高湿 0.4；
误差约 ±0.5，对 GHI 影响 < 4%，用于云量反演足够。

**SCS-CN 在山洪模型中的角色**：仅作为 `f_runoff` 修正项（限幅 0.9–1.3），
主项仍是“多窗口雨量 / 本地临界雨量”的线性比值，保证结果可解释、等级不抖动。

**月下中天**：取月中天后 12 h 25 min（半个太阴日），比“+12 h”更贴近真实潮汐周期。

### 4.3 明确不移植

| 项目 | 原因 |
|---|---|
| flow-forecast（LSTM 洪水预报） | 需要海量历史径流训练，车机无算力/无标签；改为云端训练后下发本地阈值 |
| Wflow.jl / AGCLSTM 等水文模型 | 依赖 DEM、河道断面、土壤参数，且为 GPL/重型 Julia 栈 |
| SolunarPeriods、Suntimes（GPL-3.0） | 许可证不允许闭源分发，只参考算法思想，自研实现 |

---

## 5. 三模型细则

### 5.1 山洪（1–5 级）

```
ratio = max(r1/TH1h, r3/TH3h, r6/TH6h, r24/TH24h)
s = ratio × f_api × f_soil × f_slope × f_runoff
等级：s<0.45→1, <0.75→2, <1.0→3, <1.35→4, ≥1.35→5
```

- `f_api = 1 + min(0.35, API/120)`，API 为前 24 h 雨量按 `K=0.87` 指数衰减累加；
- `f_soil`：土壤湿度 ≥0.36 → 1.2；≤0.15 → 0.85；否则 1.0；
- `f_slope = 0.9 + 0.4·min(1, tanθ/0.7)`；
- `f_runoff` 由 SCS-CN 的 24 h 产流系数偏离基准 0.35 的程度给出。
- 概率：对数正态 + `erf` CDF，`σ = 0.12 + 0.018·提前时效`，归一化到 5 级。

**临界雨量**：Open-Meteo Archive(ERA5) 近一年逐时降水，取 1/3/6 h 的 98 分位、
24 h 的 95 分位，并与经验下限（8/15/20/40 mm）取大；失败则用默认值并标注 `source=default`。

### 5.2 露营（0–100）

```
score = (0.42·体感 + 0.22·降水 + 0.16·风 + 0.12·露点差 + 0.08·UV) × 100
```

- 体感用 UTCI；`22℃` 附近高斯衰减（σ=11）；
- 硬否决：阵风 ≥15 m/s、暴雨、CAPE≥1000 且降水概率≥60%、体感 ≤−13 或 ≥38；
  **出现否决时总分封顶 45**，避免“81 分但不建议”的矛盾；
- 结论：有否决 → 不建议；≥65 适宜；≥45 需谨慎；否则不建议；
- 连续窗口：逐时 score ≥60 且无否决。

### 5.3 钓鱼（0–100）

```
weather = (0.25·气压 + 0.15·水温 + 0.15·风速 + 0.10·云量 + 0.05·降水) / 0.70
score   = (0.40·weather + 0.55·Solunar + 0.05·低光) × 100
```

- 气压项用 24 h 变化 `Δp`：缓升/高压更好，`p_score = clamp(0.5 − Δp/8, 0.15, 1)`；
- 水温为一阶滞后模型：`Tw += 0.03·(Ta − Tw)`，初值取过去 72 h 气温均值；
- Solunar 强度：命中主期 1.0、次期 0.65，否则基础 0.30；
- 窗口条件：连续 ≥2 h 且逐时 score ≥65，且无 `降水≥6 mm/h` / `阵风≥17 m/s`；
- 低光加成覆盖夜间、日出后 2 h、日落前 2 h。

---

## 6. 数据源与字段

Open-Meteo 逐时字段（`sources.HOURLY_FIELDS`）：

```
temperature_2m, relative_humidity_2m, dew_point_2m,
precipitation, precipitation_probability, pressure_msl, surface_pressure,
wind_speed_10m, wind_gusts_10m, wind_direction_10m, cloud_cover,
shortwave_radiation, uv_index, cape, soil_moisture_0_to_1cm, visibility
```

- 预报：过去 1 天 + 未来 3 天（`past_days=1, forecast_days=3`）；
- 历史：`archive-api` 近 365 天逐时降水（仅用于临界雨量分位）；
- 高程：`elevation` 接口，一次批量 9 点用于坡度差分；
- 本地 CSV 列：`time,mode,t_c,rh_pct,p_sta_hpa,rain_mm,lux,speed_kmh`。

---

## 7. 使用

```bash
# 出发前（在线）
python3 car_weather.py predict --lat 30.25 --lon 119.72 --eta 2026-10-30T18:00+08:00

# 出发前（离线合成数据，结果可复现）
python3 car_weather.py predict --lat 30.25 --lon 119.72 --eta 2026-10-30T18:00+08:00 --offline

# 生成驻车传感器样例（含 +0.6℃ 系统偏差，用于演示闭环）
python3 car_weather.py sample --out samples/sensor_parked.csv --minutes 100

# 抵达驻车后校正
python3 car_weather.py correct --csv samples/sensor_parked.csv --lat 30.25 --lon 119.72
```

公共参数：`--stay-hours`（默认 12）、`--out`（落盘 JSON，与 stdout 一致）、
`--summary`（stderr 输出人读摘要）。

### 输出契约（两阶段同构）

```jsonc
{
  "meta": {"stage": "pre_departure|post_arrival", "generated_at": "...",
           "dest": [lat, lon], "eta": "...", "stay_hours": 12,
           "data_source": "open-meteo|synthetic|...+sensor",
           "elevation_m": 36, "slope_deg": 0.3, "confidence": 0.85,
           "needs_recheck_before": "..."},
  "flood":   {"level": 1, "label": "极低", "prob": [0.99, 0.01, 0, 0, 0],
              "peak_time": "...", "onset_time": null, "score_ratio": 0.01,
              "runoff_24h_mm": 0.0, "curve_number": 75,
              "drivers": ["坡度 0°", "土壤湿度 0.31 m³/m³"],
              "thresholds": {"1h": 8, "3h": 15, "6h": 20, "24h": 40},
              "threshold_source": "archive-era5|default"},
  "camping": {"score": 73, "verdict": "适宜", "veto": null, "window": "...",
              "reasons": ["体感 13（utci）", "..."], "min_temp": 13.8,
              "max_gust": 6.8, "condensation_hours": 0},
  "fishing": {"score": 92, "windows": ["..."], "best_time": "...",
              "drivers": ["命中 Solunar 主期", "..."],
              "solunar_note": "Meeus 月历 + GMST，窗口中天/月出月落 ±1h"},
  "actions": ["...", "..."]
}
```

`correct` 额外字段：

```jsonc
{
  "observed":    {"time": "...", "t": 23.7, "rh": 62, "p_sta": 1015.9,
                  "rain_1h": 0, "lux": 36078, "cloud_from_lux": 0.24,
                  "span_min": 95, "n_valid": 96, "qc_flags": []},
  "corrections": {"assimilation_weight": 0.75,
                  "instant_bias_vs_forecast": {"t": 0.69, "rh": 1.2, "p": -0.1},
                  "note": "instant_bias 仅供入库做长期校准，不直接回灌本次结果"},
  "delta":       {"flood_level": 0, "camping_score": 0, "fishing_score": 0},
  "baseline":    {"flood_level": 1, "camping_score": 73, "fishing_score": 92}
}
```

---

## 8. 性能与验证

| 指标 | 结果 | 说明 |
|---|---|---|
| 离线 `predict` 全链路 | **9 ms** | 含 96 点合成序列、UTCI、Solunar、三模型 |
| `utci` 单次 | **30 µs** | 210 项、幂表查值，无 `**` 重复运算 |
| `moon_times` 单次 | **1.9 ms** | 145 个 10 min 采样点；Solunar 结果按地点/日期缓存 |
| 窗口求和 | O(1)/窗口 | 前缀和替代切片求和 |
| 网络请求 | 3 次/预测 | 预报、9 点高程、历史降水（历史可失败降级） |

数值校验：

- UTCI：3000 组随机输入 vs pythermalcomfort 原函数，最大绝对偏差 **8.7e-12**；
  文档示例 `utci(25, 25, 1.0, 50) = 24.6`（“无热应激”）。
- 天文：vs astral，日出 **±1.6 min**、日落 **±1.2 min**、月出 **±2 min**、月落 **±3 s**。
- 晴空：Kasten-Young 地平线大气质量 **37.92**；天顶 GHI≈1022 W/m²（tl=3, 海平面）。
- 同化：注入 +0.6℃ 系统偏差，`instant_bias.t≈0.69` 可被识别。

---

## 9. 已知限制与后续方向

1. **Tmrt 为经验近似**：只有短波辐照输入，无长波/表面温度；后续可接入天气预报的
   `direct_normal_irradiance` + 地表温度，或用车顶黑球温度计。
2. **Linke 浑浊度用气候代理**：若车机存储允许，替换为 SoDa 月平均数据可将 GHI 误差压到 <1%。
3. **坡度由 9 点高程差分**：0.01° 网格对通信基站/城市峡谷不足；后续可下发本地 DEM 瓦片。
4. **临界雨量是统计分位**：非水文模型率定值；建议云端按流域做阈值回归后增量下发。
5. **合成数据仅用于演示/回归**：`data_source=synthetic` 时置信度自动 −0.15，HMI 必须提示。
6. **后向兼容**：`car_weather.py` 保留原 CLI；包内 API 以 `predict/correct` 为稳定契约，
   其余模块可自由重构。

---

## 10. 运行环境

- Python 3.9+（实测 3.14.8），**运行期零第三方依赖**；
- 可选：有网络时访问 Open-Meteo 公共 API（无需 Key）；
- 目录可只读（除 `--out` / `sample` 指定路径）。
