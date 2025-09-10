import streamlit as st

# TODO: ALL TEMP; use session state later
from lib.simulation import Cannon, Radar
from lib.utils import Vector

# TODO:
# have a constants .py folder that defines all default states.
# and also a function (?) that spams:
# st.session_state["..."] = st.session_state["..."] if "..." in st.session_state else DEFAULT_...
# for all settings.

# DEFAULT: CANNON
df_cannon_pos: Vector = Vector(1, 3, 5)
df_cannon_velocity: int = 120
df_cannon_length: int = 12
df_cannon_g: float = 0.12
df_cannon_cd: float = 0.11
df_cannon_yaw: float = 0.0
df_cannon_pitch: float = 0.0
df_cannon_min_pitch: float = -30.0
df_cannon_max_pitch: float = 60.0
df_cannon_max_length: int = 128  # not in Cannon(), but related.

# DEFAULT: TARGET_POS
default_target_pos: Vector = Vector(6, 7, 8)

# DEFAULT: RADAR
default_radar_pos: Vector = default_target_pos
default_radar_range: int = 123
default_radar_scan_rate: int = 1
default_radar_drop_rate: float = 0.2

# DEFAULT: ESTIMATOR
# default cd and g are reused from cannon
df_assumed_velocity_range: tuple[int, int] = (10, 100)
df_max_assumed_velocity: int = 1000

# DEFAULT: ENVIRONMENT
df_max_environment_size: int = 1500

# DEFAULT: MISC
default_trajectory_type: str = "high"  # "low" or "high"
default_fire_at_target: bool = True
default_perform_estimation: bool = True

# INITIAL: CANNON
cannon: Cannon = Cannon(
    df_cannon_pos,
    df_cannon_velocity,
    df_cannon_length,
    df_cannon_g,
    df_cannon_cd,
    df_cannon_yaw,
    df_cannon_pitch,
    df_cannon_min_pitch,
    df_cannon_max_pitch,
)

# INITIAL: TARGET_POS
target_pos: Vector = default_target_pos

# INITIAL: RADAR
radar: Radar = Radar(
    default_radar_pos,
    default_radar_range,
    default_radar_scan_rate,
    default_radar_drop_rate,
)

# INITIAL: ESTIMATOR
assumed_velocity_range: tuple[int, int] = df_assumed_velocity_range

# INITIAL: ENVIRONMENT
environment_shape: dict[str, tuple[int, int]] = dict(
    x=(-df_max_environment_size, df_max_environment_size),
    y=(-df_max_environment_size, df_max_environment_size),
    z=(-df_max_environment_size, df_max_environment_size),
)

# INITIAL: MISC
trajectory_type: str = default_trajectory_type
fire_at_target: bool = default_fire_at_target
perform_estimation: bool = default_perform_estimation


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


def sb_clc_target():
    st.text("Target")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            "",
            value=target_pos.x,
            min_value=environment_shape["x"][0],
            max_value=environment_shape["x"][1],
            key="x_target",
            label_visibility="collapsed",
        )
        col[1].number_input(
            "",
            value=target_pos.y,
            min_value=environment_shape["y"][0],
            max_value=environment_shape["y"][1],
            key="y_target",
            label_visibility="collapsed",
        )
        col[2].number_input(
            "",
            value=target_pos.z,
            min_value=environment_shape["z"][0],
            max_value=environment_shape["z"][1],
            key="z_target",
            label_visibility="collapsed",
        )


def sb_clc_cannon():
    st.text("Cannon")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            "",
            value=cannon.pos.x,
            min_value=environment_shape["x"][0],
            max_value=environment_shape["x"][1],
            key="x_cannon",
            label_visibility="collapsed",
        )
        col[1].number_input(
            "",
            value=cannon.pos.y,
            min_value=environment_shape["y"][0],
            max_value=environment_shape["y"][1],
            key="y_cannon",
            label_visibility="collapsed",
        )
        col[2].number_input(
            "",
            value=cannon.pos.z,
            min_value=environment_shape["z"][0],
            max_value=environment_shape["z"][1],
            key="z_cannon",
            label_visibility="collapsed",
        )
        st.number_input(
            "Muzzle velocity (m/s)",
            value=cannon.v_ms,
            min_value=1,
            step=10,
            placeholder=f"{df_cannon_velocity}",
            help="For big cannons, 1 charge is +40 m/s.",
        )
        st.number_input(
            "Cannon length",
            value=cannon.length,
            min_value=1,
            max_value=df_cannon_max_length,
            placeholder=f"{df_cannon_length}",
            help="24 is max nethersteel big cannon length.",
        )
        st.select_slider(
            "Trajectory type",
            value=trajectory_type,
            options=("low", "high"),
            help="Artillery guns usually take the high trajectory.",
        )
        st.slider(
            "Pitch range",
            value=(cannon.min_pitch, cannon.max_pitch),
            min_value=-90.0,
            max_value=90.0,
            step=1.0,
            format="%d",
            help="Relevant for vertically built cannons or CBC addon cannons.",
        )
        st.number_input(
            "Drag",
            value=cannon.c_d,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{df_cannon_cd}",
            help=f"{df_cannon_cd} for big cannons.",
        )
        st.number_input(
            "Gravity",
            value=cannon.g,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{df_cannon_g}",
            help=f"{df_cannon_g} for big cannons.",
        )
        st.toggle(
            "Fire at target",
            value=fire_at_target,
            help="Disable if you want to manually fire at ... somewhere.",
        )
        st.number_input(
            "Yaw",
            value=cannon.yaw,
            min_value=-180.0,
            max_value=180.0,
            step=0.1,
            placeholder=f"{df_cannon_yaw}",
            disabled=not fire_at_target,
            help="-180 to 180",
        )
        st.number_input(
            "Pitch",
            value=cannon.pitch,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            placeholder=f"{df_cannon_pitch}",
            disabled=not fire_at_target,
            help="-90 (down) to +90 (up).",
        )


def sb_rev_cannon(disable: bool):
    st.text(
        "Assumed cannon stats",
        help="If known, highly recommend to set these. It will improve estimator consistency massively.",
    )
    with st.container(border=True, gap=None):
        st.number_input(
            "Drag",
            value=None,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{df_cannon_cd}",
            help=f"{df_cannon_cd} for big cannons.",
            disabled=disable,
        )
        st.number_input(
            "Gravity",
            value=None,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{df_cannon_g}",
            help=f"{df_cannon_g} for big cannons.",
            disabled=disable,
        )
        st.slider(
            "Velocity range (m/s)",
            value=assumed_velocity_range,
            min_value=1,
            max_value=df_max_assumed_velocity,
            step=1,
            help=f"{df_assumed_velocity_range} for big cannons, helps estimator prune bogus results.",
            disabled=disable,
        )


def sb_rev_radar(disable: bool):
    st.text("Radar")
    with st.container(border=True):
        cl = st.columns(3)
        center_text(cl, ["X", "Y", "Z"])
        cl[0].number_input(
            "",
            value=None,
            min_value=environment_shape["x"][0],
            max_value=environment_shape["x"][1],
            placeholder=f"{target_pos.x}",
            key="x_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        cl[1].number_input(
            "",
            value=None,
            min_value=environment_shape["y"][0],
            max_value=environment_shape["y"][1],
            placeholder=f"{target_pos.y}",
            key="y_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        cl[2].number_input(
            "",
            value=None,
            min_value=environment_shape["z"][0],
            max_value=environment_shape["z"][1],
            placeholder=f"{target_pos.z}",
            key="z_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        st.number_input(
            "Range",
            value=radar.range,
            min_value=1,
            # max_value should be min of absolute val of all area vals... a pain.
            step=1,
            placeholder=f"{default_radar_range}",
            help="Radar scan radius in blocks.",
            disabled=disable,
        )
        st.number_input(
            "Scan rate",
            value=radar.scan_rate,
            min_value=1,
            step=1,
            placeholder=f"{default_radar_scan_rate}",
            help="The radar scans once every N ticks.",
            disabled=disable,
        )
        st.number_input(
            "Drop rate",
            value=radar.drop_rate,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            placeholder=f"{default_radar_drop_rate}",
            help="Proportion of skipped observations to simulate lag.",
            disabled=disable,
        )


def sb_environment():
    st.text("Maximum size", help="Usually there's no need to touch this.")
    with st.container(border=True):
        st.slider(
            "X",
            value=environment_shape["x"],
            min_value=-df_max_environment_size,
            max_value=df_max_environment_size,
        )
        st.slider(
            "Y",
            value=environment_shape["y"],
            min_value=-df_max_environment_size,
            max_value=df_max_environment_size,
        )
        st.slider(
            "Z",
            value=environment_shape["z"],
            min_value=-df_max_environment_size,
            max_value=df_max_environment_size,
        )


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
            sb_clc_target()
            sb_clc_cannon()

        with st.expander("Reverse calculator"):
            st.toggle(
                "Estimate muzzle",
                value=perform_estimation,
                help="Disabling this causes the settings below to have no impact.",
            )
            st.button(
                "Reroll observations",
                icon="🔃",
                help="Radar drop rate involves randomness and you can get unlucky, causing the estimator to fail.",
                disabled=not perform_estimation,
            )
            sb_rev_cannon(not perform_estimation)
            sb_rev_radar(not perform_estimation)

        with st.expander("Environment", expanded=False):
            sb_environment()

        sb_credits()


def plot(fig):
    theme = None if st.context.theme.type == "light" else "streamlit"
    with st.container(border=True, gap=None):
        st.plotly_chart(fig, use_container_width=True, theme=theme)
        if theme is not None:
            st.toast("Apologies for the ugly dark plot theme.", duration="short")


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
