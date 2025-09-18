import plotly.graph_objects as go
import numpy as np
from time import time

# from lib.simulation import Cannon, Radar, perform_simulation
from lib.simulation import *
from lib.utils import Vector

CANNON = Cannon(Vector(50, 10, 80), 260, 24, 0.05, 0.99, 0, 0, -30, 60)
TARGET_POS = Vector(800, 10, 340)
RADAR = Radar(TARGET_POS, 450, 5, 0.2)

yaw, pitch, t = calculate_yaw_pitch_t(CANNON, TARGET_POS, False)
CANNON.yaw = yaw
CANNON.pitch = pitch

yr, pr = math.radians(CANNON.yaw), math.radians(CANNON.pitch)
apos = CANNON.pos.add(
    Vector(math.cos(pr) * -math.sin(yr), math.sin(pr), math.cos(pr) * math.cos(yr)).mul(
        CANNON.length
    )
)

obs = []
while len(obs) <= 2:
    obs = get_observed_trajectory(simulate_trajectory(CANNON, round(t)), RADAR)

iterations = 1000

fails = 0
t1 = time()
for i in range(iterations):
    stats = estimate_muzzle2(obs)
    if stats is None:
        fails += 1
        continue
    if stats.get("pos").sub(apos).length() > 0.001:
        fails += 1

t2 = time()
print("Solver-step hybrid.")
print(f"{iterations - fails}/{iterations}")
print(f"Time taken: {t2 - t1}")

fails = 0
t1 = time()
for i in range(iterations):
    stats = estimate_muzzle(obs)
    if stats is None:
        fails += 1
        continue
    if stats.get("pos").sub(apos).length() > 0.001:
        fails += 1

t2 = time()
print("Full simulation")
print(f"{iterations - fails}/{iterations}")
print(f"Time taken: {t2 - t1}")
