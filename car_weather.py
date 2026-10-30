#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""car_weather.py — 兼容入口，实际实现位于 ``carweather`` 包。

    python3 car_weather.py predict --lat 30.25 --lon 119.72 --eta 2026-10-30T18:00+08:00
    python3 car_weather.py sample  --out samples/sensor_parked.csv
    python3 car_weather.py correct --csv samples/sensor_parked.csv --lat 30.25 --lon 119.72
"""

import sys

from carweather.cli import main

if __name__ == "__main__":
    sys.exit(main())
