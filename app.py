import streamlit as st

from lib.plotting import *
from lib.simulation import *
from lib.ui import *
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


@st.fragment
def get_stats():
    return perform_simulation(
        cannon=st.session_state["cannon"],
        target_pos=st.session_state["target_pos"],
        fire_at_target=st.session_state["fire_at_target"],
        trajectory_type=st.session_state["trajectory_type"],
        perform_estimation=st.session_state["perform_estimation"],
        radar=st.session_state["radar"],
        assumed_cd=st.session_state["assumed_c_d"],
        assumed_g=st.session_state["assumed_g"],
        assumed_v_ms_range=st.session_state["assumed_velocity_range"],
    )

@st.fragment
def generate_fig(stats):
    return populate_plot(
        fig=init_plot(st.session_state["environment_shape"], height=600),
        cannon=st.session_state["cannon"],
        target_pos=st.session_state["target_pos"],
        radar=st.session_state["radar"],
        stats=stats,
    )


sidebar()
t1, t2, t3 = st.tabs(["Plot", "Results", "Debug"])
stats = get_stats()
with t1:
    plot(generate_fig(stats))
with t2:
    results(stats)
with t3:
    st.write(st.session_state)

# TODO: perform caching and session state and/or fragment
# TODO: fix yaw (urgent) inconsistency
