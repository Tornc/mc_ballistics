import streamlit as st

# TODO: ALL TEMP; use session state later
from lib.constr import *
from lib.utils import safe_extract


def center_text(columns, texts):
    for col, text in zip(columns, texts):
        col.markdown(f'<div class="centered-text">{text}</div>', unsafe_allow_html=True)


def set_clc_target():
    st.text("Target")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        st.session_state["target_pos"].x = col[0].number_input(
            "",
            value=st.session_state["target_pos"].x,
            min_value=st.session_state["environment_shape"]["x"][0],
            max_value=st.session_state["environment_shape"]["x"][1],
            key="x_target",
            label_visibility="collapsed",
        )
        st.session_state["target_pos"].y = col[1].number_input(
            "",
            value=st.session_state["target_pos"].y,
            min_value=st.session_state["environment_shape"]["y"][0],
            max_value=st.session_state["environment_shape"]["y"][1],
            key="y_target",
            label_visibility="collapsed",
        )
        st.session_state["target_pos"].z = col[2].number_input(
            "",
            value=st.session_state["target_pos"].z,
            min_value=st.session_state["environment_shape"]["z"][0],
            max_value=st.session_state["environment_shape"]["z"][1],
            key="z_target",
            label_visibility="collapsed",
        )


def set_clc_cannon():
    st.text("Cannon")
    with st.container(border=True):
        col = st.columns(3)
        center_text(col, ["X", "Y", "Z"])
        st.session_state["cannon"].pos.x = col[0].number_input(
            "",
            value=st.session_state["cannon"].pos.x,
            min_value=st.session_state["environment_shape"]["x"][0],
            max_value=st.session_state["environment_shape"]["x"][1],
            key="x_cannon",
            label_visibility="collapsed",
        )
        st.session_state["cannon"].pos.y = col[1].number_input(
            "",
            value=st.session_state["cannon"].pos.y,
            min_value=st.session_state["environment_shape"]["y"][0],
            max_value=st.session_state["environment_shape"]["y"][1],
            key="y_cannon",
            label_visibility="collapsed",
        )
        st.session_state["cannon"].pos.z = col[2].number_input(
            "",
            value=st.session_state["cannon"].pos.z,
            min_value=st.session_state["environment_shape"]["z"][0],
            max_value=st.session_state["environment_shape"]["z"][1],
            key="z_cannon",
            label_visibility="collapsed",
        )
        st.session_state["cannon"].v_ms = st.number_input(
            "Muzzle velocity (m/s)",
            value=st.session_state["cannon"].v_ms,
            min_value=1,
            step=10,
            placeholder=f"{DF_CANNON_VELOCITY}",
            help="For big cannons, 1 charge is +40 m/s.",
        )
        st.session_state["cannon"].length = st.number_input(
            "Cannon length",
            value=st.session_state["cannon"].length,
            min_value=1,
            max_value=DF_CANNON_MAX_LENGTH,
            placeholder=f"{DF_CANNON_LENGTH}",
            help="24 is max nethersteel big cannon length.",
        )
        st.session_state["trajectory_type"] = st.select_slider(
            "Trajectory type",
            value=st.session_state["trajectory_type"],
            options=("low", "high"),
            help="Artillery guns usually take the high trajectory.",
        )
        st.session_state["cannon"].min_pitch, st.session_state["cannon"].max_pitch = (
            st.slider(
                "Pitch range",
                value=(
                    st.session_state["cannon"].min_pitch,
                    st.session_state["cannon"].max_pitch,
                ),
                min_value=-90.0,
                max_value=90.0,
                step=1.0,
                format="%d",
                help="Relevant for vertically built cannons or CBC addon cannons.",
            )
        )
        st.session_state["cannon"].c_d = st.number_input(
            "Drag",
            value=st.session_state["cannon"].c_d,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_CD}",
            help=f"{DF_CANNON_CD} for big cannons.",
        )
        st.session_state["cannon"].g = st.number_input(
            "Gravity",
            value=st.session_state["cannon"].g,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_G}",
            help=f"{DF_CANNON_G} for big cannons.",
        )
        st.session_state["fire_at_target"] = st.toggle(
            "Fire at target",
            value=st.session_state["fire_at_target"],
            help="Disable if you want to manually fire at ... somewhere.",
        )
        st.session_state["cannon"].yaw = st.number_input(
            "Yaw",
            value=st.session_state["cannon"].yaw,
            min_value=-180.0,
            max_value=180.0,
            step=0.1,
            placeholder=f"{DF_CANNON_YAW}",
            disabled=not st.session_state["fire_at_target"],
            help="-180 to 180",
        )
        st.session_state["cannon"].pitch = st.number_input(
            "Pitch",
            value=st.session_state["cannon"].pitch,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            placeholder=f"{DF_CANNON_PITCH}",
            disabled=not st.session_state["fire_at_target"],
            help="-90 (down) to +90 (up).",
        )


def set_rev_cannon(disable: bool):
    st.text(
        "Assumed cannon stats",
        help="If known, highly recommend to set these. It will improve estimator consistency massively.",
    )
    with st.container(border=True, gap=None):
        st.session_state["assumed_c_d"] = st.number_input(
            "Drag coefficient",
            value=st.session_state["assumed_c_d"],
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_CD}",
            help=f"{DF_CANNON_CD} for big cannons.",
            disabled=disable,
        )
        st.session_state["assumed_g"] = st.number_input(
            "Gravity",
            value=st.session_state["assumed_g"],
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            placeholder=f"{DF_CANNON_G}",
            help=f"{DF_CANNON_G} for big cannons.",
            disabled=disable,
        )
        st.session_state["assumed_velocity_range"] = st.slider(
            "Velocity range (m/s)",
            value=st.session_state["assumed_velocity_range"],
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
        st.session_state["radar"].pos.x = cl[0].number_input(
            "",
            value=st.session_state["radar"].pos.x,
            min_value=st.session_state["environment_shape"]["x"][0],
            max_value=st.session_state["environment_shape"]["x"][1],
            placeholder=f"{st.session_state["target_pos"].x}",
            key="x_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        st.session_state["radar"].pos.y = cl[1].number_input(
            "",
            value=st.session_state["radar"].pos.y,
            min_value=st.session_state["environment_shape"]["y"][0],
            max_value=st.session_state["environment_shape"]["y"][1],
            placeholder=f"{st.session_state["target_pos"].y}",
            key="y_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        st.session_state["radar"].pos.z = cl[2].number_input(
            "",
            value=st.session_state["radar"].pos.z,
            min_value=st.session_state["environment_shape"]["z"][0],
            max_value=st.session_state["environment_shape"]["z"][1],
            placeholder=f"{st.session_state["target_pos"].z}",
            key="z_radar",
            label_visibility="collapsed",
            disabled=disable,
        )
        st.session_state["radar"].range = st.number_input(
            "Range",
            value=st.session_state["radar"].range,
            min_value=1,
            # max_value should be min of absolute val of all area vals... a pain.
            step=1,
            placeholder=f"{DF_RADAR_RANGE}",
            help="Radar scan radius in blocks.",
            disabled=disable,
        )
        st.session_state["radar"].scan_rate = st.number_input(
            "Scan rate",
            value=st.session_state["radar"].scan_rate,
            min_value=1,
            step=1,
            placeholder=f"{DF_RADAR_SCAN_RATE}",
            help="The radar scans once every N ticks.",
            disabled=disable,
        )
        st.session_state["radar"].drop_rate = st.number_input(
            "Drop rate",
            value=st.session_state["radar"].drop_rate,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            placeholder=f"{DF_RADAR_DROP_RATE}",
            help="Proportion of skipped observations to simulate lag.",
            disabled=disable,
        )


# def set_environment():
#     st.text("Maximum size", help="Usually there's no need to touch this.")
#     with st.container(border=True):
#         st.session_state["environment_shape"]["x"] = st.slider(
#             "X",
#             value=st.session_state["environment_shape"]["x"],
#             min_value=-DF_MAX_ENVIRONMENT_SIZE,
#             max_value=DF_MAX_ENVIRONMENT_SIZE,
#         )
#         st.session_state["environment_shape"]["y"] = st.slider(
#             "Y",
#             value=st.session_state["environment_shape"]["y"],
#             min_value=-DF_MAX_ENVIRONMENT_SIZE,
#             max_value=DF_MAX_ENVIRONMENT_SIZE,
#         )
#         st.session_state["environment_shape"]["z"] = st.slider(
#             "Z",
#             value=st.session_state["environment_shape"]["z"],
#             min_value=-DF_MAX_ENVIRONMENT_SIZE,
#             max_value=DF_MAX_ENVIRONMENT_SIZE,
#         )


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
            st.session_state["perform_estimation"] = st.toggle(
                "Estimate muzzle",
                value=st.session_state["perform_estimation"],
                help="Disabling this causes the settings below to have no impact.",
            )
            # TODO: implement!!!
            st.button(
                "Reroll observations TODO: implement",
                icon="🔃",
                help="Radar drop rate involves randomness and you can get unlucky, causing the estimator to fail.",
                disabled=not st.session_state["perform_estimation"],
            )
            set_rev_cannon(not st.session_state["perform_estimation"])
            set_rev_radar(not st.session_state["perform_estimation"])

        # with st.expander("Environment", expanded=False):
        #     set_environment()

        set_credits()


def plot(fig):
    # TODO: better night theme
    theme = None if st.context.theme.type == "light" else "streamlit"
    st.plotly_chart(fig, use_container_width=True, theme=theme)


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
    st.text(
        "Estimator", help="These values are derived from samples obtained by radar."
    )

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
