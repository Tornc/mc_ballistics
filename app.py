import streamlit as st

from lib.constr import *
from lib.plotting import *
from lib.simulation import *
from lib.ui import *
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


@st.fragment
def generate_statistics():
    st.session_state["cannon"] = Cannon(
        pos=Vector(
            ssg("cannon_pos_x"),
            ssg("cannon_pos_y"),
            ssg("cannon_pos_z"),
        ),
        v_ms=ssg("cannon_velocity"),
        length=ssg("cannon_length"),
        g=ssg("cannon_g"),
        cd=ssg("cannon_cd"),
        yaw=ssg("cannon_yaw"),
        pitch=ssg("cannon_pitch"),
        min_pitch=ssg("cannon_pitch_range")[0],
        max_pitch=ssg("cannon_pitch_range")[1],
    )
    st.session_state["target_pos"] = Vector(
        ssg("target_pos_x"),
        ssg("target_pos_y"),
        ssg("target_pos_z"),
    )
    st.session_state["radar"] = Radar(
        pos=Vector(
            (
                ssg("radar_pos_x")
                if ssg("radar_pos_x") is not None
                else ssg("target_pos_x")
            ),
            (
                ssg("radar_pos_y")
                if ssg("radar_pos_y") is not None
                else ssg("target_pos_y")
            ),
            (
                ssg("radar_pos_z")
                if ssg("radar_pos_z") is not None
                else ssg("target_pos_z")
            ),
        ),
        range=ssg("radar_range"),
        scan_rate=ssg("radar_scan_rate"),
        drop_rate=ssg("radar_drop_rate"),
    )
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
    )


@st.fragment
def generate_figure():
    return populate_plot(
        fig=init_plot(DF_MAX_ENVIRONMENT_SIZE, height=600),
        cannon=ssg("cannon"),
        target_pos=ssg("target_pos"),
        radar=ssg("radar"),
        display_radar_range=ssg("perform_estimation"),
        stats=ssg("statistics"),
    )


init_state()
sidebar()
generate_statistics()

t1, t2 = st.tabs(
    [
        "Plot",
        "Results",
        # "Debug",
    ]
)
with t1:
    plot(generate_figure())
with t2:
    results()
# with t3:
#     st.write(st.session_state)
