import plotly.graph_objects as go
import numpy as np
from time import time
from lib.simulation import Cannon, Radar, perform_simulation
from lib.utils import Vector

CANNON = Cannon(Vector(50, 10, 80), 320, 24, 0.05, 0.99, 0, 0, -30, 60)
TARGET_POS = Vector(800, 10, 340)
RADAR = Radar(TARGET_POS, 250, 5, 0.2)


def results():
    stats = perform_simulation(
        CANNON,
        TARGET_POS,
        True,
        "high",
        True,
        RADAR,
        # assumed_cd=0.99,
        # assumed_g=0.05,
        # assumed_v_ms_range=(40, 320),
        # assumed_v_ms_multiple=40,
    )
    if stats.get("n_obs") > 2 and stats.get("est_muzzle_pos") is None:
        return True
    # print(f"n_obs: {stats.get("n_obs")}")
    # print(f"est_muzzle_pos: {stats.get("est_muzzle_pos")}")
    # emp = stats.get("error_est_muzzle_pos")
    # print(f"error_est_muzzle_pos: {round(emp, 5) if emp is not None else emp}")
    # print(f"est_v_ms: {stats.get("est_v_ms")}")
    # print(f"est_cd: {stats.get("est_cd")}")
    # print(f"est_g: {stats.get("est_g")}")


cts, its = 0, 100
t1 = time()
for i in range(0, its):
    if results():
        cts += 1
t2 = time()
print()
print(f"{its-cts}/{its}")
print(t2 - t1, (t2 - t1) / its)
