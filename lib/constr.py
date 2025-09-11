# constants x streamlit get it 😂😂😂😂😂😂😂😂😂😂😂😂😂
# sorry

from lib.utils import Vector

# DEFAULT: CANNON
DF_CANNON_POS: Vector = Vector(1, 3, 5)
DF_CANNON_VELOCITY: int = 120
DF_CANNON_LENGTH: int = 12
DF_CANNON_G: float = 0.12
DF_CANNON_CD: float = 0.11
DF_CANNON_YAW: float = 0.0
DF_CANNON_PITCH: float = 0.0
DF_CANNON_MIN_PITCH: float = -30.0
DF_CANNON_MAX_PITCH: float = 60.0
DF_CANNON_MAX_LENGTH: int = 128  # not in Cannon(), but related.

# DEFAULT: TARGET_POS
DF_TARGET_POS: Vector = Vector(6, 7, 8)

# DEFAULT: RADAR
DF_RADAR_POS: Vector = DF_TARGET_POS
DF_RADAR_RANGE: int = 123
DF_RADAR_SCAN_RATE: int = 1
DF_RADAR_DROP_RATE: float = 0.2

# DEFAULT: ESTIMATOR
# default cd and g are reused from cannon
DF_ASSUMED_VELOCITY_RANGE: tuple[int, int] = (10, 100)
DF_MAX_ASSUMED_VELOCITY: int = 1000

# DEFAULT: ENVIRONMENT
DF_MAX_ENVIRONMENT_SIZE: int = 1500

# DEFAULT: MISC
DF_FIRE_AT_TARGET: bool = True
DF_TRAJECTORY_TYPE: str = "high"  # "low" or "high"
DF_PERFORM_ESTIMATION: bool = True

