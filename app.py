import cython as c
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.plotting import *
from lib.simulation import *
from lib.utils import *

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

pitch_range = (-30.0, 60.0)
CANNON = Cannon(Vector(50, 10, 80), 280, 24, 0.05, 0.99, 0, 0)
radar = Radar(Vector(0, 0, 0), 250, 0.25, 0.0)
target_pos = Vector(800, 10, 340)
radar.pos = target_pos

yaw, pitch, t = calculate_yaw_pitch_t(CANNON, target_pos, low=True)
if yaw is not None and pitch is not None and t is not None:
    CANNON.yaw, CANNON.pitch = yaw, pitch
# TODO: make UI actually functional

environment_size = dict(x=(-1500, 1500), y=(-1500, 1500), z=(-1500, 1500))

fig = init_plot(environment_size, height=600)
fig = generate_plot(fig, CANNON, radar, target_pos, round(t))


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


@st.fragment
def balloon_summoner():
    cl = st.columns([0.25, 0.45, 0.3])
    cl[0].text("By Tornc")
    cl[1].markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Tornc/mc_ballistics)"
    )
    cl[2].markdown(
        "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)"
    )
    btn = cl[0].button("🎈", help="Very professional.")
    if btn:
        st.balloons()


# TODO: move parts to ui.py or split even further.
with st.sidebar:
    with st.expander("Calculator", expanded=True):
        st.text("Cannon")
        with st.container(border=True):
            col = st.columns(3)
            center_text(col, ["X", "Y", "Z"])
            col[0].number_input(
                "",
                value=CANNON.pos.x,
                min_value=environment_size["x"][0],
                max_value=environment_size["x"][1],
                key="x_cannon",
                label_visibility="collapsed",
            )
            col[1].number_input(
                "",
                value=CANNON.pos.y,
                min_value=environment_size["y"][0],
                max_value=environment_size["y"][1],
                key="y_cannon",
                label_visibility="collapsed",
            )
            col[2].number_input(
                "",
                value=CANNON.pos.z,
                min_value=environment_size["z"][0],
                max_value=environment_size["z"][1],
                key="z_cannon",
                label_visibility="collapsed",
            )
            st.number_input(
                "Muzzle velocity (m/s)",
                min_value=1,
                value=320,
                step=10,
                placeholder="320",
                help="For big cannons, 1 charge is +40 m/s.",
            )
            st.number_input(
                "Cannon length",
                min_value=1,
                max_value=512,  # should be enough
                value=24,
                placeholder="24",
                help="24 is max nethersteel big cannon length.",
            )
            st.select_slider(
                "Trajectory type",
                ("low", "high"),
                "high",
                help="Artillery guns usually take the high trajectory.",
            )
            st.number_input(
                "Drag",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                placeholder="0.01",
                help="0.01 for big cannons.",
            )
            st.number_input(
                "Gravity",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                placeholder="0.05",
                help="0.05 for big cannons.",
            )
            e_cannon_temp = st.toggle(
                "Fire at target",
                value=True,
                help="Disable if you want to manually fire at ... somewhere.",
            )
            st.number_input(
                "Yaw",
                min_value=-180.0,
                max_value=180.0,
                value=float(CANNON.yaw),
                step=0.1,
                placeholder="0.0",
                disabled=e_cannon_temp,
                help="-180 to 180",
            )
            st.number_input(
                "Pitch",
                min_value=pitch_range[0],
                max_value=pitch_range[1],
                value=float(CANNON.pitch),
                step=0.1,
                placeholder="0.0",
                disabled=e_cannon_temp,
                help="-90 (down) to +90 (up).",
            )
            pitch_range = st.slider(
                "Pitch range",
                -90,
                90,
                (-30, 60),
                step=1,
                help="Relevant for vertically built cannons or CBC addon cannons.",
            )

        st.text("Target")
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            "",
            value=CANNON.pos.x,
            min_value=environment_size["x"][0],
            max_value=environment_size["x"][1],
            key="x_target",
            label_visibility="collapsed",
        )
        col[1].number_input(
            "",
            value=CANNON.pos.y,
            min_value=environment_size["y"][0],
            max_value=environment_size["y"][1],
            key="y_target",
            label_visibility="collapsed",
        )
        col[2].number_input(
            "",
            value=CANNON.pos.z,
            min_value=environment_size["z"][0],
            max_value=environment_size["z"][1],
            key="z_target",
            label_visibility="collapsed",
        )

    with st.expander("Reverse calculator"):
        e_perform_estimation = st.toggle(
            "Estimate muzzle",
            value=True,
            help="Disabling this causes the settings below to have no impact.",
        )
        st.text("Assumed cannon stats")
        with st.container(border=True, gap=None):
            e_assumed_drag = st.number_input(
                "Drag",
                min_value=0.0,
                max_value=1.0,
                value=None,
                step=0.01,
                placeholder="0.01",
                help="0.01 for big cannons.",
            )
            e_assumed_gravity = st.number_input(
                "Gravity",
                min_value=0.0,
                max_value=1.0,
                value=None,
                step=0.01,
                placeholder="0.05",
                help="0.05 for big cannons.",
            )
            e_assumed_length = st.number_input(
                "Cannon length",
                min_value=1,
                max_value=512,  # should be enough
                value=None,
                placeholder="24",
                help="24 is max nethersteel big cannon length.",
            )

        st.text("Radar stats")
        with st.container(border=True):
            col = st.columns(3)
            center_text(col, ["X", "Y", "Z"])
            col[0].number_input(
                "",
                value=None,
                min_value=environment_size["x"][0],
                max_value=environment_size["x"][1],
                placeholder=f"{target_pos.x}",
                key="x_radar",
                label_visibility="collapsed",
            )
            col[1].number_input(
                "",
                value=None,
                min_value=environment_size["y"][0],
                max_value=environment_size["y"][1],
                placeholder=f"{target_pos.y}",
                key="y_radar",
                label_visibility="collapsed",
            )
            col[2].number_input(
                "",
                value=None,
                min_value=environment_size["z"][0],
                max_value=environment_size["z"][1],
                placeholder=f"{target_pos.z}",
                key="z_radar",
                label_visibility="collapsed",
            )
            e_radar_range = st.number_input(
                "Range",
                value=250,
                min_value=1,
                # max_value should be min of absolute val of all area vals... a pain.
                step=1,
                placeholder="250",
                help="Radar scan radius in blocks.",
            )
            # TODO: snap to 0.05 steps
            e_radar_scanrate = st.number_input(
                "Scanrate",
                min_value=1,
                value=1,
                step=1,
                placeholder="1",
                help="The radar scans once every N ticks.",
            )
            e_radar_droprate = st.number_input(
                "Droprate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                placeholder="0.1",
                help="Proportion of skipped observations to simulate lag.",
            )
            reroll = st.button(
                "Regenerate observations",
                icon="🔃",
                help="Drop-rate involves randomness and you can get unlucky, causing the estimator to fail.",
            )

    with st.expander("Environment", expanded=False):
        st.text("Maximum size", help="Usually there's no need to touch this.")
        with st.container(border=True):
            e_x_range = st.slider("X", -1500, 1500, (-1500, 1500))
            e_y_range = st.slider("Y", -1500, 1500, (-1500, 1500))
            e_z_range = st.slider("Z", -1500, 1500, (-1500, 1500))

    balloon_summoner()


st.plotly_chart(fig, use_container_width=True, theme=None)


with st.expander("Info", True):
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.write("Cannon")
    with c2.container(border=True):
        st.write("Estimator")

# This is good for indicating failed observation or sth
# st.toast("yo", icon=None, duration=3)

# TODO: perform caching and session state and/or fragment
# TODO: fix yaw (urgent) inconsistency
