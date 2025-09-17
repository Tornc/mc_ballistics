import plotly.graph_objects as go
import numpy as np
from time import time

# from lib.simulation import Cannon, Radar, perform_simulation
from lib.simulation import *
from lib.utils import Vector

CANNON = Cannon(Vector(50, 10, 80), 260, 24, 0.05, 0.99, 0, 0, -30, 60)
TARGET_POS = Vector(800, 10, 340)
# RADAR = Radar(TARGET_POS, 250, 5, 0.2)
RADAR = Radar(TARGET_POS, 250, 1, 0.2)

yaw, pitch, t = calculate_yaw_pitch_t(CANNON, TARGET_POS, True)
CANNON.yaw = yaw
CANNON.pitch = pitch

yr, pr = math.radians(CANNON.yaw), math.radians(CANNON.pitch)
print("Actual muzzle pos:")
print(
    CANNON.pos.add(
        Vector(
            math.cos(pr) * -math.sin(yr), math.sin(pr), math.cos(pr) * math.cos(yr)
        ).mul(CANNON.length)
    ).round(0.001)
)
obs = get_observed_trajectory(simulate_trajectory(CANNON, round(t)), RADAR)
stats = estimate_muzzle2(
    observations=obs, min_v=None, max_v=None, vms_multiple=None, cd=None, g=None
)
print(stats.get("pos").round(0.001))
