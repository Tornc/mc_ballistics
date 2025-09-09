import cython as c
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.plotting import *
from lib.simulation import *
from lib.utils import *
from lib.ui import sidebar, results

ICON_PATH = "./assets/img/icon.ico"

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

CANNON = Cannon(Vector(50, 10, 80), 280, 24, 0.05, 0.99, 0, 0, -30.0, 60.0)
radar = Radar(Vector(0, 0, 0), 250, 0.25, 0.0)
target_pos = Vector(800, 10, 340)
radar.pos = target_pos

yaw, pitch, t = calculate_yaw_pitch_t(CANNON, target_pos, low=True)
if yaw is not None and pitch is not None and t is not None:
    CANNON.yaw, CANNON.pitch = yaw, pitch
# TODO: make UI actually functional

env_dims = dict(x=(-1500, 1500), y=(-1500, 1500), z=(-1500, 1500))


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


fig = init_plot(env_dims, height=600)
fig = generate_plot(fig, CANNON, radar, target_pos, round(t))


# TODO: move parts to ui.py or split even further.
sidebar()
with st.container(border=True, gap=None):
    st.plotly_chart(fig, use_container_width=True, theme=None)
results()

# This is good for indicating failed observation or sth
# st.toast("yo", icon=None, duration=3)

# TODO: perform caching and session state and/or fragment
# TODO: fix yaw (urgent) inconsistency
