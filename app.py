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

cannon = Cannon(Vector(50, 10, 80), 280, 24, 0.05, 0.99, 0, 0, -30.0, 60.0)
radar = Radar(Vector(0, 0, 0), 250, 1, 0.0)
target_pos = Vector(800, 10, 340)
radar.pos = target_pos

stats = perform_simulation(
    cannon, target_pos, True, "high", True, radar, None, None, None
)

env_dims = dict(x=(-1500, 1500), y=(-1500, 1500), z=(-1500, 1500))
fig = init_plot(env_dims, height=600)
fig = populate_plot(fig, cannon, target_pos, radar, stats)

sidebar()
t1, t2 = st.tabs(["Plot", "Results"])
with t1:
    plot(fig)
with t2:
    results(stats)


# This is good for indicating failed observation or sth
# st.toast("yo", icon=None, duration=3)

# TODO: perform caching and session state and/or fragment
# TODO: fix yaw (urgent) inconsistency
