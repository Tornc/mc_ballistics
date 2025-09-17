import streamlit as st
from lib.constr import *
from lib.simulation import perform_simulation
from lib.utils import ssg


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


def sett_clc_target():
    st.text("Target")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            key="target_pos_x",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )
        col[1].number_input(
            key="target_pos_y",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )
        col[2].number_input(
            key="target_pos_z",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )


def sett_clc_cannon():
    st.text("Cannon")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        col[0].number_input(
            key="cannon_pos_x",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )
        col[1].number_input(
            key="cannon_pos_y",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )
        col[2].number_input(
            key="cannon_pos_z",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
        )
        st.number_input(
            help="For big cannons, 1 charge is +40 m/s.",
            key="cannon_velocity",
            label="Muzzle velocity (m/s)",
            min_value=1,
            placeholder=f"{DF_CANNON_VELOCITY}",
            step=10,
        )
        st.number_input(
            help="24 is max nethersteel big cannon length.",
            key="cannon_length",
            label="Cannon length",
            max_value=DF_CANNON_MAX_LENGTH,
            min_value=1,
            placeholder=f"{DF_CANNON_LENGTH}",
        )
        st.select_slider(
            help=f"Artillery guns usually take the {DF_TRAJECTORY_TYPE} trajectory.",
            key="trajectory_type",
            label="Trajectory type",
            options=("low", "high"),
        )
        st.slider(
            format="%d",
            help="Only applies to non-manual fire.",
            key="cannon_pitch_range",
            label="Pitch range",
            max_value=90.0,
            min_value=-90.0,
            step=1.0,
        )
        st.number_input(
            help=f"{DF_CANNON_CD} for big cannons.",
            key="cannon_cd",
            label="Drag",
            max_value=1.0,
            min_value=0.0,
            placeholder=f"{DF_CANNON_CD}",
            step=0.01,
        )
        st.number_input(
            help=f"{DF_CANNON_G} for big cannons.",
            key="cannon_g",
            label="Gravity",
            max_value=1.0,
            min_value=0.0,
            placeholder=f"{DF_CANNON_G}",
            step=0.01,
        )
        st.toggle(
            help="Enable if you want to manually fire at ... somewhere.",
            key="manual_fire",
            label="Manual fire",
        )
        st.slider(
            disabled=not ssg("manual_fire"),
            help="0/360 (S), 90 (W), 180 (N), 270 (E)",
            key="cannon_yaw",
            label="Yaw",
            max_value=360.0,
            min_value=0.0,
            step=0.1,
        )
        st.slider(
            disabled=not ssg("manual_fire"),
            help="-90 (down), +90 (up)",
            key="cannon_pitch",
            label="Pitch",
            max_value=90.0,
            min_value=-90.0,
            step=0.1,
        )


def sett_rev_cannon(disable: bool):
    st.text(
        "Assumed cannon stats",
        help="Leave empty if unknown, but reliability (and speed) will take a hit.",
    )
    with st.container(border=True):
        st.number_input(
            disabled=disable,
            help=f"{DF_CANNON_CD} for big cannons.",
            key="assumed_cd",
            label="Drag coefficient",
            max_value=1.0,
            min_value=0.0,
            placeholder=f"{DF_CANNON_CD}",
            step=0.01,
            value=None,
        )
        st.number_input(
            disabled=disable,
            help=f"{DF_CANNON_G} for big cannons.",
            key="assumed_g",
            label="Gravity",
            max_value=1.0,
            min_value=0.0,
            placeholder=f"{DF_CANNON_G}",
            step=0.01,
            value=None,
        )
        st.number_input(
            disabled=disable,
            help=f"{DF_ASSUMED_VELOCITY_MULTIPLE} for big cannons.",
            key="assumed_velocity_multiple",
            label="Smallest velocity multiple (m/s)",
            max_value=DF_MAX_ASSUMED_VELOCITY,
            min_value=1,
            placeholder=f"{DF_ASSUMED_VELOCITY_MULTIPLE}",
            step=10,
            value=None,
        )
        # TODO: might want to adjust min max according to assumed_velocity_multiple
        st.slider(
            disabled=disable,
            help=f"{DF_ASSUMED_VELOCITY_RANGE} for big cannons, speeds up estimator massively.",
            key="assumed_velocity_range",
            label="Velocity range (m/s)",
            max_value=DF_MAX_ASSUMED_VELOCITY,
            min_value=1,
            step=1,
        )


def sett_rev_radar(disable: bool):
    st.text("Radar")
    with st.container(border=True):
        cl = st.columns(3)
        center_text(cl, ["X", "Y", "Z"])
        cl[0].number_input(
            disabled=disable,
            key="radar_pos_x",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            placeholder=f"{ssg("target_pos_x")}",
            value=None,
        )
        cl[1].number_input(
            disabled=disable,
            key="radar_pos_y",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            placeholder=f"{ssg("target_pos_y")}",
            value=None,
        )
        cl[2].number_input(
            disabled=disable,
            key="radar_pos_z",
            label_visibility="collapsed",
            label="",
            max_value=DF_MAX_ENVIRONMENT_SIZE,
            min_value=-DF_MAX_ENVIRONMENT_SIZE,
            placeholder=f"{ssg("target_pos_z")}",
            value=None,
        )
        st.number_input(
            disabled=disable,
            help="Radar scan radius in blocks.",
            key="radar_range",
            label="Range",
            # max_value should be min of absolute val of all area vals... a pain.
            min_value=1,
            placeholder=f"{DF_RADAR_RANGE}",
            step=1,
        )
        st.number_input(
            disabled=disable,
            help="The radar scans once every N ticks.",
            key="radar_scan_rate",
            label="Scan rate",
            min_value=1,
            placeholder=f"{DF_RADAR_SCAN_RATE}",
            step=1,
        )
        st.number_input(
            disabled=disable,
            help="Proportion of skipped observations to simulate lag.",
            key="radar_drop_rate",
            label="Drop rate",
            max_value=1.0,
            min_value=0.0,
            placeholder=f"{DF_RADAR_DROP_RATE}",
            step=0.05,
        )


def sett_credits():
    cl = st.columns([0.3, 0.37, 0.25])
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
            sett_clc_target()
            sett_clc_cannon()

        with st.expander("Reverse calculator"):
            st.toggle(
                help="Disabling this causes the settings below to have no impact.",
                key="perform_estimation",
                label="Estimate muzzle",
            )
            do_reroll = st.button(
                disabled=not ssg("perform_estimation"),
                help="Radar drop rate involves randomness and you can get unlucky, causing the estimator to fail.",
                icon="🔃",
                label="Reroll observations",
            )
            if do_reroll:
                st.session_state["statistics"] = perform_simulation(
                    cannon=ssg("cannon"),
                    target_pos=ssg("target_pos"),
                    fire_at_target=not ssg("manual_fire"),
                    trajectory_type=ssg("trajectory_type"),
                    perform_estimation=ssg("perform_estimation"),
                    radar=ssg("radar"),
                    assumed_cd=ssg("assumed_cd"),
                    assumed_g=ssg("assumed_g"),
                    assumed_v_ms_range=ssg("assumed_velocity_range"),
                    assumed_v_ms_multiple=ssg("assumed_velocity_multiple"),
                )
            sett_rev_cannon(not ssg("perform_estimation"))
            sett_rev_radar(not ssg("perform_estimation"))

        sett_credits()


def plot(fig):
    theme_type = getattr(getattr(st.context, "theme", None), "type", None)
    theme = None if theme_type == "light" else "streamlit"
    st.plotly_chart(fig, use_container_width=True, theme=theme)


def res_cannon():
    st.text("Cannon")

    stats: dict = ssg("statistics")
    # I hate this.
    yaw = stats.get("yaw")
    pitch = stats.get("pitch")
    flight_time = stats.get("flight_time")
    muzzle_pos = stats.get("muzzle_pos")
    impact_pos = stats.get("impact_pos")
    error_impact = stats.get("error_impact")

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


def res_reverse():
    st.text(
        "Estimator", help="These values are derived from samples obtained by the radar."
    )

    stats: dict = ssg("statistics")
    # I still hate this.
    n_obs = stats.get("n_obs")
    est_muzzle_pos = stats.get("est_muzzle_pos")
    error_est_muzzle_pos = stats.get("error_est_muzzle_pos")
    est_t = stats.get("est_t")
    est_v_ms = stats.get("est_v_ms")
    est_cd = stats.get("est_cd")
    est_g = stats.get("est_g")
    est_yaw = stats.get("est_yaw")
    est_pitch = stats.get("est_pitch")

    est_muzzle_pos = (
        est_muzzle_pos.round().tostring() if est_muzzle_pos is not None else None
    )
    error_est_muzzle_pos = (
        round(error_est_muzzle_pos, 2) if error_est_muzzle_pos is not None else None
    )
    est_yaw = round(est_yaw, 2) if est_yaw is not None else None
    est_pitch = round(est_pitch, 2) if est_pitch is not None else None

    data = dict(
        n_obs=n_obs,
        est_muzzle_pos=est_muzzle_pos,
        error_est_muzzle_pos=error_est_muzzle_pos,
        est_v_ms=est_v_ms,
        est_cd=est_cd,
        est_g=est_g,
        est_yaw=est_yaw,
        est_pitch=est_pitch,
    )
    data = {
        "\\# of observations": n_obs,
        "Muzzle position": est_muzzle_pos,
        "Muzzle position error": error_est_muzzle_pos,
        "Flight time at 1st observation": est_t,
        "Shell velocity": est_v_ms,
        "Drag coefficient": est_cd,
        "Gravity": est_g,
        "Yaw": est_yaw,
        "Pitch": est_pitch,
    }
    for key in data:
        if data[key] is None:
            data[key] = "N/A"

    st.table(data=data)


def results():
    res_cannon()
    res_reverse()
