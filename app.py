import cython as c
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.plotting import *
from lib.simulation import *
from lib.ui import sidebar, plot, results
from lib.utils import *

ICON_PATH = "./assets/img/icon.ico"

DEFAULT_CD = 0.11
DEFAULT_G = 0.12
DEFAULT_LENGTH = 12
DEFAULT_VELOCITY = 120
DEFAULT_MIN_VELOCITY = 40
DEFAULT_MAX_VELOCITY = 320

st.set_page_config(
    page_title="Ballistics Simulation",
    page_icon=ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.centered-text {
    text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

cannon = Cannon(Vector(50, 10, 80), 280, 24, 0.05, 0.99, 0, 0, -30.0, 60.0)
radar = Radar(Vector(0, 0, 0), 250, 0.5, 0.0)
target_pos = Vector(800, 10, 340)
radar.pos = target_pos

yaw, pitch, t = calculate_yaw_pitch_t(cannon, target_pos, low=True)
if yaw is not None and pitch is not None and t is not None:
    cannon.yaw, cannon.pitch = yaw, pitch


# TODO: draw radar if pos neq target pos
def generate_plot(
    fig: go.Figure,
    cannon: Cannon,
    radar: Radar,
    target_pos: Vector,
    max_ticks: int,
):
    trajectory = simulate_trajectory(cannon, max_ticks)
    add_trace(fig, [p for _, p in trajectory], "lines", "Trajectory", 8)
    observed_trajectory = get_observed_trajectory(trajectory, radar)
    # TODO: fix weird edge case that bypasses this check somehow
    if len(observed_trajectory) > 0:
        print(len(observed_trajectory))
        add_trace(fig, [p for _, p in observed_trajectory], "markers", "Observation", 4)
        est_muzzle_pos, info = estimate_muzzle(observed_trajectory)
        if est_muzzle_pos is not None:
            print(f"Est: { est_muzzle_pos}")
            add_trace(fig, est_muzzle_pos, "markers+text", "Muzzle (estimate)")

    add_trace(fig, cannon.pos, "markers+text", "Cannon", 12)
    add_trace(fig, target_pos, "markers+text", "Target", 12)
    add_trace(fig, [cannon.pos, trajectory[0][1]], "lines", "Barrel", 8)
    add_hemisphere(fig, radar.pos, radar.range, "Radar range")
    return fig


env_dims = dict(x=(-1500, 1500), y=(-1500, 1500), z=(-1500, 1500))
fig = init_plot(env_dims, height=600)
fig = generate_plot(fig, cannon, radar, target_pos, round(t))

sidebar()
plot(fig)
results()


# This is good for indicating failed observation or sth
# st.toast("yo", icon=None, duration=3)

# TODO: perform caching and session state and/or fragment
# TODO: fix yaw (urgent) inconsistency
