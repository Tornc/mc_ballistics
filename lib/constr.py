# constants x streamlit get it 😂😂😂😂😂😂😂😂😂😂😂😂😂
# sorry
import streamlit as st

# DEFAULT: CANNON
DF_CANNON_POS_X: int = 50
DF_CANNON_POS_Y: int = 10
DF_CANNON_POS_Z: int = 80
DF_CANNON_VELOCITY: int = 320
DF_CANNON_LENGTH: int = 24
DF_CANNON_G: float = 0.05
DF_CANNON_CD: float = 0.99
DF_CANNON_YAW: float = 0.0
DF_CANNON_PITCH: float = 0.0
DF_CANNON_PITCH_RANGE: tuple[float, float] = (-30.0, 60, 0)
DF_CANNON_MAX_LENGTH: int = 128  # not in Cannon(), but related.

# DEFAULT: TARGET_POS
DF_TARGET_POS_X: int = 800
DF_TARGET_POS_Y: int = 10
DF_TARGET_POS_Z: int = 340

# DEFAULT: RADAR
DF_RADAR_POS_X: int = None
DF_RADAR_POS_Y: int = None
DF_RADAR_POS_Z: int = None
DF_RADAR_RANGE: int = 250
DF_RADAR_SCAN_RATE: int = 2
DF_RADAR_DROP_RATE: float = 0.2

# DEFAULT: ESTIMATOR
# TODO: change this later. None is for testing purposes.
# DF_ASSUMED_G = DF_CANNON_G
# DF_ASSUMED_CD = DF_CANNON_CD
DF_ASSUMED_G = None
DF_ASSUMED_CD = None
DF_ASSUMED_VELOCITY_RANGE: tuple[int, int] = (40, 320)
DF_MAX_ASSUMED_VELOCITY: int = 1000

# DEFAULT: ENVIRONMENT
DF_MAX_ENVIRONMENT_SIZE: int = 1500

# DEFAULT: MISC
DF_MANUAL_FIRE: bool = False
DF_TRAJECTORY_TYPE: str = "high"  # "low" or "high"
DF_PERFORM_ESTIMATION: bool = True


def sset(key: str, default):
    # This makes life easier.
    st.session_state[key] = (
        st.session_state[key] if key in st.session_state else default
    )


# INITIAL: CANNON
sset("cannon_pos_x", DF_CANNON_POS_X)
sset("cannon_pos_y", DF_CANNON_POS_Y)
sset("cannon_pos_z", DF_CANNON_POS_Z)
sset("cannon_velocity", DF_CANNON_VELOCITY)
sset("cannon_length", DF_CANNON_LENGTH)
sset("cannon_cd", DF_CANNON_CD)
sset("cannon_g", DF_CANNON_G)
sset("cannon_yaw", DF_CANNON_YAW)
sset("cannon_pitch", DF_CANNON_PITCH)
sset("cannon_pitch_range", DF_CANNON_PITCH_RANGE)

# INITIAL: TARGET_POS
sset("target_pos_x", DF_TARGET_POS_X)
sset("target_pos_y", DF_TARGET_POS_Y)
sset("target_pos_z", DF_TARGET_POS_Z)

# INITIAL: RADAR
sset("radar_pos_x", DF_RADAR_POS_X)
sset("radar_pos_y", DF_RADAR_POS_Y)
sset("radar_pos_z", DF_RADAR_POS_Z)
sset("radar_range", DF_RADAR_RANGE)
sset("radar_scan_rate", DF_RADAR_SCAN_RATE)
sset("radar_drop_rate", DF_RADAR_DROP_RATE)

# INITIAL: ESTIMATOR
sset("assumed_cd", DF_ASSUMED_CD)
sset("assumed_g", DF_ASSUMED_G)
sset("assumed_velocity_range", DF_ASSUMED_VELOCITY_RANGE)

# INITIAL: MISC
sset("manual_fire", DF_MANUAL_FIRE)
sset("trajectory_type", DF_TRAJECTORY_TYPE)
sset("perform_estimation", DF_PERFORM_ESTIMATION)

# ---------------------

# INITIAL: STATE
sset("cannon", None)
sset("target_pos", None)
sset("radar", None)
sset("statistics", None)
