import streamlit as st


DEFAULT_CD = 0.11
DEFAULT_G = 0.12
DEFAULT_LENGTH = 12
DEFAULT_VELOCITY = 120
DEFAULT_MIN_VELOCITY = 40
DEFAULT_MAX_VELOCITY = 320

# TODO: ALL TEMP; use session state later
from lib.simulation import Cannon
from lib.utils import Vector

cannon = Cannon(Vector(0, 0, 0), 0, 0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0)
env_size = dict(x=(-10, 10), y=(-10, 10), z=(-10, 10))
target_pos = Vector(1, 2, 3)


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


def sb_clc_cannon():
    st.text("Cannon")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            "",
            value=cannon.pos.x,
            min_value=env_size["x"][0],
            max_value=env_size["x"][1],
            key="x_cannon",
            label_visibility="collapsed",
        )
        col[1].number_input(
            "",
            value=cannon.pos.y,
            min_value=env_size["y"][0],
            max_value=env_size["y"][1],
            key="y_cannon",
            label_visibility="collapsed",
        )
        col[2].number_input(
            "",
            value=cannon.pos.z,
            min_value=env_size["z"][0],
            max_value=env_size["z"][1],
            key="z_cannon",
            label_visibility="collapsed",
        )
        st.number_input(
            "Muzzle velocity (m/s)",
            min_value=1,
            value=DEFAULT_VELOCITY,
            step=10,
            placeholder=f"{DEFAULT_VELOCITY}",
            help="For big cannons, 1 charge is +40 m/s.",
        )
        st.number_input(
            "Cannon length",
            min_value=1,
            max_value=128,  # should be enough
            value=DEFAULT_LENGTH,
            placeholder=f"{DEFAULT_LENGTH}",
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
            value=DEFAULT_CD,
            step=0.01,
            placeholder=f"{DEFAULT_CD}",
            help=f"{DEFAULT_CD} for big cannons.",
        )
        st.number_input(
            "Gravity",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_G,
            step=0.01,
            placeholder=f"{DEFAULT_G}",
            help=f"{DEFAULT_G} for big cannons.",
        )
        st.slider(
            "Pitch range",
            -90.0,
            90.0,
            (cannon.min_pitch, cannon.max_pitch),
            step=1.0,
            format="%d",
            help="Relevant for vertically built cannons or CBC addon cannons.",
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
            value=cannon.yaw,
            step=0.1,
            placeholder="0.0",
            disabled=e_cannon_temp,
            help="-180 to 180",
        )
        st.number_input(
            "Pitch",
            min_value=cannon.min_pitch,
            max_value=cannon.max_pitch,
            value=cannon.pitch,
            step=0.1,
            placeholder="0.0",
            disabled=e_cannon_temp,
            help="-90 (down) to +90 (up).",
        )


def sb_clc_target():
    st.text("Target")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            "",
            value=cannon.pos.x,
            min_value=env_size["x"][0],
            max_value=env_size["x"][1],
            key="x_target",
            label_visibility="collapsed",
        )
        col[1].number_input(
            "",
            value=cannon.pos.y,
            min_value=env_size["y"][0],
            max_value=env_size["y"][1],
            key="y_target",
            label_visibility="collapsed",
        )
        col[2].number_input(
            "",
            value=cannon.pos.z,
            min_value=env_size["z"][0],
            max_value=env_size["z"][1],
            key="z_target",
            label_visibility="collapsed",
        )


def sb_rev_cannon(disable: bool):
    st.text(
        "Assumed cannon stats",
        help="If known, highly recommend to set these. It will improve estimator consistency massively.",
    )
    with st.container(border=True, gap=None):
        st.number_input(
            "Drag",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder=f"{DEFAULT_CD}",
            help=f"{DEFAULT_CD} for big cannons.",
            disabled=disable,
        )
        st.number_input(
            "Gravity",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder=f"{DEFAULT_G}",
            help=f"{DEFAULT_G} for big cannons.",
            disabled=disable,
        )
        st.slider(
            "Velocity range (m/s)",
            1,
            1000,
            (DEFAULT_MIN_VELOCITY, DEFAULT_MAX_VELOCITY),
            step=1,
            help=f"{DEFAULT_MIN_VELOCITY}-{DEFAULT_MAX_VELOCITY} for big cannons, helps estimator prune bogus results.",
        )


def sb_rev_radar(disable: bool):
    st.text("Radar")
    with st.container(border=True):
        cl = st.columns(3)
        center_text(cl, ["X", "Y", "Z"])
        cl[0].number_input(
            "",
            value=None,
            min_value=env_size["x"][0],
            max_value=env_size["x"][1],
            placeholder=f"{target_pos.x}",
            key="x_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        cl[1].number_input(
            "",
            value=None,
            min_value=env_size["y"][0],
            max_value=env_size["y"][1],
            placeholder=f"{target_pos.y}",
            key="y_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        cl[2].number_input(
            "",
            value=None,
            min_value=env_size["z"][0],
            max_value=env_size["z"][1],
            placeholder=f"{target_pos.z}",
            key="z_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        st.number_input(
            "Range",
            value=250,
            min_value=1,
            # max_value should be min of absolute val of all area vals... a pain.
            step=1,
            placeholder="250",
            help="Radar scan radius in blocks.",
            disabled=disable,
        )
        st.number_input(
            "Scanrate",
            min_value=1,
            value=1,
            step=1,
            placeholder="1",
            help="The radar scans once every N ticks.",
            disabled=disable,
        )
        st.number_input(
            "Drop rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            placeholder="0.1",
            help="Proportion of skipped observations to simulate lag.",
            disabled=disable,
        )


def sb_environment_size():
    st.text("Maximum size", help="Usually there's no need to touch this.")
    with st.container(border=True):
        st.slider("X", -1500, 1500, (-1500, 1500))
        st.slider("Y", -1500, 1500, (-1500, 1500))
        st.slider("Z", -1500, 1500, (-1500, 1500))


def sb_credits():
    cl = st.columns([0.3, 0.4, 0.3], gap=None)
    cl[0].text("Made with ❤️")
    cl[1].markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Tornc/mc_ballistics)"
    )
    cl[2].markdown(
        "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)"
    )


def sidebar():
    with st.sidebar:
        with st.expander("Calculator", expanded=False):
            sb_clc_cannon()
            sb_clc_target()

        with st.expander("Reverse calculator"):
            e_perform_estimation = st.toggle(
                "Estimate muzzle",
                value=True,
                help="Disabling this causes the settings below to have no impact.",
            )
            st.button(
                "Reroll observations",
                icon="🔃",
                help="Radar drop rate involves randomness and you can get unlucky, causing the estimator to fail.",
                disabled=not e_perform_estimation,
            )
            sb_rev_cannon(not e_perform_estimation)
            sb_rev_radar(not e_perform_estimation)

        with st.expander("Environment", expanded=False):
            sb_environment_size()

        sb_credits()


def rs_cannon():
    st.write("Cannon")


def rs_reverse():
    st.write("Estimator")


def results():
    with st.expander("Results", True):
        c1, c2 = st.columns(2, border=True)
        with c1:
            rs_cannon()
        with c2:
            rs_reverse()
