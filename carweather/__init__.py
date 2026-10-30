# -*- coding: utf-8 -*-
"""carweather — 车载 5 传感器 + Open-Meteo 的三任务决策包。

模块划分
--------
* ``common``   时区/修约/窗口和/统计基础函数
* ``astro``    太阳与月球位置、升落、Solunar 窗口
* ``physics``  UTCI、Ineichen 晴空、大气质量、SCS-CN 产流
* ``sources``  Open-Meteo 与本地传感器接入（含离线合成数据）
* ``models``   山洪 / 露营 / 钓鱼三个决策模型
* ``pipeline`` 两段式流程（predict → correct）
* ``cli``      命令行入口

对外只暴露高层接口，内部模块可单独测试。
"""

from .pipeline import correct, predict
from .sources import make_sample

__version__ = "1.0.0"

__all__ = ["predict", "correct", "make_sample", "__version__"]
