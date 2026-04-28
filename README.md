# DAARS: Distance-Aware Adaptive Reward Shaping for Safe Robot Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)

---

## Overview

DAARS replaces fixed reward weights in safe navigation with **distance-dependent scaling functions** that dynamically modulate progress and safety components based on obstacle proximity. Near obstacles, progress drive is suppressed while safety penalty is amplified; in open space, the reward degenerates to standard fixed-weight shaping.