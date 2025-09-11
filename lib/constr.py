# constants x streamlit get it 😂😂😂😂😂😂😂😂😂😂😂😂😂
# sorry

import streamlit as st
from lib.simulation import Cannon, Radar
from lib.utils import Vector

# DEFAULT: CANNON
DF_CANNON_POS: Vector = Vector(50, 10, 80)
DF_CANNON_VELOCITY: int = 320
DF_CANNON_LENGTH: int = 24
DF_CANNON_G: float = 0.05
DF_CANNON_CD: float = 0.99
DF_CANNON_YAW: float = 0.0
DF_CANNON_PITCH: float = 0.0
DF_CANNON_MIN_PITCH: float = -30.0
DF_CANNON_MAX_PITCH: float = 60.0
DF_CANNON_MAX_LENGTH: int = 128  # not in Cannon(), but related.

# DEFAULT: TARGET_POS
DF_TARGET_POS: Vector = Vector(800, 10, 340)

# DEFAULT: RADAR
DF_RADAR_POS: Vector = DF_TARGET_POS
DF_RADAR_RANGE: int = 250
DF_RADAR_SCAN_RATE: int = 1
DF_RADAR_DROP_RATE: float = 0.2

# DEFAULT: ESTIMATOR
# default cd and g are reused from cannon
DF_ASSUMED_VELOCITY_RANGE: tuple[int, int] = (40, 320)
DF_MAX_ASSUMED_VELOCITY: int = 1000

# DEFAULT: ENVIRONMENT
DF_MAX_ENVIRONMENT_SIZE: int = 1500

# DEFAULT: MISC
DF_FIRE_AT_TARGET: bool = True
DF_TRAJECTORY_TYPE: str = "high"  # "low" or "high"
DF_PERFORM_ESTIMATION: bool = True

# INITIAL: CANNON
st.session_state["cannon"] = (
    st.session_state["cannon"]
    if "cannon" in st.session_state
    else Cannon(
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
)

# INITIAL: TARGET_POS
st.session_state["target_pos"] = (
    st.session_state["target_pos"]
    if "target_pos" in st.session_state
    else DF_TARGET_POS
)

# INITIAL: RADAR
st.session_state["radar"] = (
    st.session_state["radar"]
    if "radar" in st.session_state
    else Radar(
        DF_RADAR_POS,
        DF_RADAR_RANGE,
        DF_RADAR_SCAN_RATE,
        DF_RADAR_DROP_RATE,
    )
)
st.session_state["radar_pos_x"] = st.session_state["radar_pos_x"] if "radar_pos_x" in st.session_state else None
st.session_state["radar_pos_y"] = st.session_state["radar_pos_y"] if "radar_pos_y" in st.session_state else None
st.session_state["radar_pos_z"] = st.session_state["radar_pos_z"] if "radar_pos_z" in st.session_state else None

# INITIAL: ESTIMATOR
st.session_state["assumed_c_d"] = (
    st.session_state["assumed_c_d"] if "assumed_c_d" in st.session_state else None
)
st.session_state["assumed_g"] = (
    st.session_state["assumed_g"] if "assumed_g" in st.session_state else None
)
st.session_state["assumed_velocity_range"] = (
    st.session_state["assumed_velocity_range"]
    if "assumed_velocity_range" in st.session_state
    else DF_ASSUMED_VELOCITY_RANGE
)

# INITIAL: MISC
st.session_state["fire_at_target"] = (
    st.session_state["fire_at_target"]
    if "fire_at_target" in st.session_state
    else DF_FIRE_AT_TARGET
)
st.session_state["trajectory_type"] = (
    st.session_state["trajectory_type"]
    if "trajectory_type" in st.session_state
    else DF_TRAJECTORY_TYPE
)
st.session_state["perform_estimation"] = (
    st.session_state["perform_estimation"]
    if "perform_estimation" in st.session_state
    else DF_PERFORM_ESTIMATION
)


# INITIAL: STATE

st.session_state["stats"] = None