import streamlit as st

# TODO: ALL TEMP; use session state later
from lib.constr import *
from lib.simulation import Cannon, Radar
from lib.utils import *

# TODO:
# have a constants .py folder that defines all default states.
# and also a function (?) that spams:
# st.session_state["..."] = st.session_state["..."] if "..." in st.session_state else DEFAULT_...
# for all settings.

# INITIAL: CANNON
cannon: Cannon = Cannon(
    DF_CANNON_POS,
    DF_CANNON_VELOCITY,
    DF_CANNON_LENGTH,
    DF_CANNON_G,
    DF_CANNON_CD,
    DF_CANNON_YAW,
    DF_CANNON_PITCH,
    DF_CANNON_MIN_PITCH,
    DF_CANNON_MAX_PITCH,
)

# INITIAL: TARGET_POS
target_pos: Vector = DF_TARGET_POS

# INITIAL: RADAR
radar: Radar = Radar(
    DF_RADAR_POS,
    DF_RADAR_RANGE,
    DF_RADAR_SCAN_RATE,
    DF_RADAR_DROP_RATE,
)

# INITIAL: ESTIMATOR
assumed_velocity_range: tuple[int, int] = DF_ASSUMED_VELOCITY_RANGE

# INITIAL: ENVIRONMENT
environment_shape: dict[str, tuple[int, int]] = dict(
    x=(-DF_MAX_ENVIRONMENT_SIZE, DF_MAX_ENVIRONMENT_SIZE),
    y=(-DF_MAX_ENVIRONMENT_SIZE, DF_MAX_ENVIRONMENT_SIZE),
    z=(-DF_MAX_ENVIRONMENT_SIZE, DF_MAX_ENVIRONMENT_SIZE),
)

# INITIAL: MISC
fire_at_target: bool = DF_FIRE_AT_TARGET
trajectory_type: str = DF_TRAJECTORY_TYPE
perform_estimation: bool = DF_PERFORM_ESTIMATION


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


def set_clc_target():
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


def set_clc_cannon():
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
            placeholder=f"{DF_CANNON_VELOCITY}",
            help="For big cannons, 1 charge is +40 m/s.",
        )
        st.number_input(
            "Cannon length",
            value=cannon.length,
            min_value=1,
            max_value=DF_CANNON_MAX_LENGTH,
            placeholder=f"{DF_CANNON_LENGTH}",
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
            placeholder=f"{DF_CANNON_CD}",
            help=f"{DF_CANNON_CD} for big cannons.",
        )
        st.number_input(
            "Gravity",
            value=cannon.g,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_G}",
            help=f"{DF_CANNON_G} for big cannons.",
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
            placeholder=f"{DF_CANNON_YAW}",
            disabled=not fire_at_target,
            help="-180 to 180",
        )
        st.number_input(
            "Pitch",
            value=cannon.pitch,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            placeholder=f"{DF_CANNON_PITCH}",
            disabled=not fire_at_target,
            help="-90 (down) to +90 (up).",
        )


def set_rev_cannon(disable: bool):
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
            placeholder=f"{DF_CANNON_CD}",
            help=f"{DF_CANNON_CD} for big cannons.",
            disabled=disable,
        )
        st.number_input(
            "Gravity",
            value=None,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_G}",
            help=f"{DF_CANNON_G} for big cannons.",
            disabled=disable,
        )
        st.slider(
            "Velocity range (m/s)",
            value=assumed_velocity_range,
            min_value=1,
            max_value=DF_MAX_ASSUMED_VELOCITY,
            step=1,
            help=f"{DF_ASSUMED_VELOCITY_RANGE} for big cannons, helps estimator prune bogus results.",
            disabled=disable,
        )


def set_rev_radar(disable: bool):
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
            placeholder=f"{DF_RADAR_RANGE}",
            help="Radar scan radius in blocks.",
            disabled=disable,
        )
        st.number_input(
            "Scan rate",
            value=radar.scan_rate,
            min_value=1,
            step=1,
            placeholder=f"{DF_RADAR_SCAN_RATE}",
            help="The radar scans once every N ticks.",
            disabled=disable,
        )
        st.number_input(
            "Drop rate",
            value=radar.drop_rate,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            placeholder=f"{DF_RADAR_DROP_RATE}",
            help="Proportion of skipped observations to simulate lag.",
            disabled=disable,
        )


def set_environment():
    st.text("Maximum size", help="Usually there's no need to touch this.")
    with st.container(border=True):
        st.slider(
            "X",
            value=environment_shape["x"],
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            max_value=DF_MAX_ENVIRONMENT_SIZE,
        )
        st.slider(
            "Y",
            value=environment_shape["y"],
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            max_value=DF_MAX_ENVIRONMENT_SIZE,
        )
        st.slider(
            "Z",
            value=environment_shape["z"],
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            max_value=DF_MAX_ENVIRONMENT_SIZE,
        )


def set_credits():
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
            set_clc_target()
            set_clc_cannon()

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
            set_rev_cannon(not perform_estimation)
            set_rev_radar(not perform_estimation)

        with st.expander("Environment", expanded=False):
            set_environment()

        set_credits()


def plot(fig):
    theme = None if st.context.theme.type == "light" else "streamlit"
    st.plotly_chart(fig, use_container_width=True, theme=theme)
    if theme is not None:
        st.toast("Apologies for the ugly dark plot theme.", duration="short")


def rs_cannon(stats):
    st.text("Cannon")

    # I hate this.
    yaw = safe_extract("yaw", stats)
    pitch = safe_extract("pitch", stats)
    flight_time = safe_extract("flight_time", stats)
    muzzle_pos = safe_extract("muzzle_pos", stats)
    impact_pos = safe_extract("impact_pos", stats)
    error_impact = safe_extract("error_impact", stats)

    yaw = round(yaw, 2) if yaw is not None else None
    pitch = round(pitch, 2) if pitch is not None else None
    muzzle_pos = muzzle_pos.round().tostring() if muzzle_pos is not None else None
    impact_pos = impact_pos.round().tostring() if impact_pos is not None else None
    error_impact = round(error_impact, 2) if error_impact is not None else None

    data = {
        "Yaw": yaw,
        "Pitch": pitch,
        "Shell flight time": flight_time,
        "Muzzle position": muzzle_pos,
        "Impact position": impact_pos,
        "Impact to target error": error_impact,
    }
    for key in data:
        if data[key] is None:
            data[key] = "N/A"

    st.table(data=data)


def rs_reverse(stats):
    st.text("Estimator", help="These values are derived from samples obtained by radar.")

    # I still hate this.
    n_obs = safe_extract("n_obs", stats)
    est_muzzle_pos = safe_extract("est_muzzle_pos", stats)
    est_v_ms = safe_extract("est_v_ms", stats)
    est_g = safe_extract("est_g", stats)
    est_c_d = safe_extract("est_c_d", stats)
    est_yaw = safe_extract("est_yaw", stats)
    est_pitch = safe_extract("est_pitch", stats)
    error_est_muzzle_pos = safe_extract("error_est_muzzle_pos", stats)

    est_muzzle_pos = (
        est_muzzle_pos.round().tostring() if est_muzzle_pos is not None else None
    )
    est_yaw = round(est_yaw, 2) if est_yaw is not None else None
    est_pitch = round(est_pitch, 2) if est_pitch is not None else None
    error_est_muzzle_pos = (
        round(error_est_muzzle_pos, 2) if error_est_muzzle_pos is not None else None
    )

    data = dict(
        n_obs=n_obs,
        est_muzzle_pos=est_muzzle_pos,
        est_v_ms=est_v_ms,
        est_g=est_g,
        est_c_d=est_c_d,
        est_yaw=est_yaw,
        est_pitch=est_pitch,
        error_est_muzzle_pos=error_est_muzzle_pos,
    )
    data = {
        "\# of observations": n_obs,
        "Muzzle position": est_muzzle_pos,
        "Shell velocity": est_v_ms,
        "Gravity": est_g,
        "Drag coefficient": est_c_d,
        "Yaw": est_yaw,
        "Pitch": est_pitch,
        "Muzzle position error": error_est_muzzle_pos,
    }
    for key in data:
        if data[key] is None:
            data[key] = "N/A"

    st.table(data=data)


def results(stats):
    rs_cannon(stats)
    rs_reverse(stats)
