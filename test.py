from lib.plotting import *
from lib.simulation import *
from lib.ui import sidebar, plot, results
from lib.utils import *

# DEFAULT_HM = "hmmm..."

# state = dict(hi="hello!", hm="broroooo")
# state["hm"] = state["hm"] if "hm" in state else DEFAULT_HM
# a = state["hm"]

# print(a)


cannon = Cannon(Vector(50, 10, 80), 280, 24, 0.05, 0.99, 0, 0, -30.0, 60.0)
radar = Radar(Vector(0, 0, 0), 250, 1, 0.0)
target_pos = Vector(800, 10, 340)
radar.pos = target_pos

# yaw, pitch, t = calculate_yaw_pitch_t(cannon, target_pos, low=True)
# if yaw is not None and pitch is not None and t is not None:
#     cannon.yaw, cannon.pitch = yaw, pitch
#     trajectory = simulate_trajectory(cannon, round(t))
#     observed_trajectory = get_observed_trajectory(trajectory, radar)
#     if len(observed_trajectory) > 0:
#         est_muzzle_pos, info = estimate_muzzle(observed_trajectory)
#         if est_muzzle_pos is not None:
#             print(f"Est: { est_muzzle_pos}")
#             print(f"Info: {info}")
#             print(round(t))
