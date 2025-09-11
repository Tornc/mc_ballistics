import math
import random
from collections import Counter
from dataclasses import dataclass
from time import time

from lib.utils import *


@dataclass
class Cannon:
    pos: Vector
    v_ms: float
    length: int
    g: float
    c_d: float
    yaw: float
    pitch: float
    min_pitch: float
    max_pitch: float


@dataclass
class Radar:
    pos: Vector
    range: int
    scan_rate: int
    drop_rate: float


def calculate_pitch(
    distance: float,
    velocity_ms: float,
    target_height: float,
    cannon_length: int,
    gravity: float,
    drag_coefficient: float,
    low: bool,
):
    """
    All calculations come from Endal's ballistics calculator made in Desmos (https://www.desmos.com/calculator/az4angyumw),
    there may be bugs because the formulas sure look like some kind of alien language to me. It's >60x faster than
    brute-forcing pitch, and has higher precision to boot.
    """

    # Constants
    g = gravity  # CBC gravity
    c_d = drag_coefficient  # drag coefficient
    t0 = 0  # Minimum projectile flight time in ticks
    tn = 750  # Maximum projectile flight time in ticks
    start_step_size = 18.75

    # Inputs
    X_R = distance
    v_m = velocity_ms
    h = target_height
    L = cannon_length

    u = v_m / 20  # Convert to velocity per tick

    # Higher order parameters
    A = g * c_d / (u * (1 - c_d))

    def B(t):
        return t * (g * c_d / (1 - c_d)) * 1 / X_R

    C = L / (u * X_R) * (g * c_d / (1 - c_d)) + h / X_R

    # The idea is to start with very large steps and decrease step size
    # the closer we get to the actual value.
    num_halvings = 10  # This is fine, too many halvings will slow down
    acceptable_threshold = 0.001  # How close to X_R is good enough

    def a_R(t):
        # "watch out for the square root" -Endal
        B_t = B(t)
        in_root = -(A**2) + B_t**2 + C**2 + 2 * B_t * C + 1
        if in_root < 0:
            return None

        num_a_R = math.sqrt(in_root) - 1
        den_a_R = A + B_t + C
        return 2 * math.atan(num_a_R / den_a_R)

    # t = time projectile, either start from t0 and increment or tn and decrement
    t = t0 if low else tn
    increasing_t = low
    step_size = start_step_size

    a_R1 = None  # scope reasons, since we need to return it

    for _ in range(num_halvings):
        while True:
            # It's taking too long, give up
            if (low and t >= tn) or (not low and t <= t0):
                return None, None

            # Angle of projectile at t
            a_R1 = a_R(t)
            # a square root being negative means something
            # has gone wrong, so give up
            if a_R1 is None:
                return None, None

            # Distance of projectile at t
            p1_X_R1 = u * math.cos(a_R1) / math.log(c_d)
            p2_X_R1 = c_d**t - 1
            p3_X_R1 = L * math.cos(a_R1)
            X_R1 = p1_X_R1 * p2_X_R1 + p3_X_R1

            # Good enough, let's call it quits
            if abs(X_R1 - X_R) <= acceptable_threshold:
                break

            # We've passed the target (aka we're close), now oscillate around the actual
            # target value until it's 'good enough' or it's taking too long.
            if (increasing_t and X_R1 > X_R) or (not increasing_t and X_R1 < X_R):
                increasing_t = not increasing_t
                break

            t = t + (step_size if increasing_t == low else -step_size)

        # Increase the precision after breaking out, since we're closer to target
        step_size = step_size / 2

    return t, math.degrees(a_R1)


def calculate_yaw_pitch_t(cannon: Cannon, target_pos: Vector, low: bool):
    dpos = target_pos.sub(cannon.pos)
    horizontal_dist = math.sqrt(dpos.x**2 + dpos.z**2)
    t, pitch = calculate_pitch(
        distance=horizontal_dist,
        velocity_ms=cannon.v_ms,
        target_height=dpos.y,
        cannon_length=cannon.length,
        gravity=cannon.g,
        drag_coefficient=cannon.c_d,
        low=low,
    )

    if t is None:
        return None, None, None

    yaw = math.degrees(math.atan2(dpos.x, dpos.z))
    return yaw, pitch, t


def simulate_trajectory(
    cannon: Cannon, max_ticks: int = None, stop_y: int = None
) -> list[tuple[int, Vector]]:
    """
    According to @sashafiesta#1978's formula on Discord:
    Vx = 0.99 * Vx
    Vy = 0.99 * Vy - 0.05
    """
    assert (max_ticks == None) ^ (stop_y == None), "One of these must be set."

    yaw_rad = math.radians(cannon.yaw)
    pitch_rad = math.radians(cannon.pitch)

    # Cannon aiming direction
    dir = Vector(
        math.cos(pitch_rad) * math.sin(yaw_rad),
        math.sin(pitch_rad),
        math.cos(pitch_rad) * math.cos(yaw_rad),
    )
    # Muzzle (initial projectile) position
    pos = cannon.pos.add(dir.mul(cannon.length))
    # v/s -> v/t
    vel = dir.mul(cannon.v_ms / 20)

    trajectory: list[tuple[int, Vector]] = []

    if max_ticks is not None:
        for t in range(max_ticks + 1):
            trajectory.append((t, pos.copy()))
            pos = pos.add(vel)
            vel = vel.mul(cannon.c_d)
            vel.y -= cannon.g

    # TODO: termination condition is wrong here
    # NOTE: Cannon can be under or above target.
    if stop_y is not None:
        raise NotImplementedError
        for t in range(100000):  # Endless calculation protection.
            trajectory.append((t, pos.copy()))
            pos = pos.add(vel)
            vel = vel.mul(cannon.c_d)
            vel.y -= cannon.g

    return trajectory


def get_observed_trajectory(
    trajectory: list[tuple[int, Vector]], radar: Radar
) -> list[tuple[float, Vector]]:
    """
    Args:
        trajectory (list[tuple[int, Vector]]): timestamp in **ticks**, position
        radar (Radar):
        drop_rate (float, optional): Portion of measurements that get skipped range: [0, 1]. Defaults to 0.

    Returns:
        list[tuple[float, Vector]]: timestamp in **seconds**, position
    """
    t0 = time()  # Pretend this is when the radar sees projectile.
    items = []
    for t, pos in trajectory:
        if random.random() < radar.drop_rate:
            continue
        if t % radar.scan_rate != 0:
            continue
        # distance
        if pos.sub(radar.pos).length() <= radar.range:
            items.append((t0 + tick2sec(t), pos))
    return items


# TODO: clean this garbage up. (and improve it)

EPSILON_ANGLE = 1e-12
EPSILON_TIME = 1e-9
CDG_DIGITS = 5
ACCEPTABLE_RMSE = 1e-3


# TODO: maybe sec2tick here is a bad idea, because information might be lost.
def compute_obs_velocities(
    datapoints: list[tuple[float, Vector]],
) -> list[tuple[int, Vector]]:
    """
    Note that this will return list of len(datapoints) - 1, because we can't know
    the velocity of the first datapoint.

    Args:
        datapoints (list[tuple[float, Vector]]): **seconds**, position

    Returns:
        list[tuple[float, Vector]]: **ticks**, velocity
    """
    velocities = []
    for i in range(len(datapoints) - 1):
        t0, p0 = datapoints[i]
        t1, p1 = datapoints[i + 1]
        dt = sec2tick(t1 - t0)  # You never know what in-game lag will do.
        if dt <= 0:
            continue
        vel = p1.sub(p0).div(dt)
        velocities.append((dt, vel))
    return velocities


def estimate_cd(velocities: list[int, Vector]) -> float | None:
    if len(velocities) < 2:
        return None

    cd_candidates = []
    for i in range(len(velocities) - 1):
        dt, v0 = velocities[i]
        _, v1 = velocities[i + 1]

        if abs(v0.x) > EPSILON_ANGLE and v1.x * v0.x > EPSILON_ANGLE:
            cd_val = (v1.x / v0.x) ** (1 / dt)
            cd_candidates.append(round(cd_val, CDG_DIGITS))
        if abs(v0.z) > EPSILON_ANGLE and v1.z * v0.z > EPSILON_ANGLE:
            cd_val = (v1.z / v0.z) ** (1 / dt)
            cd_candidates.append(round(cd_val, CDG_DIGITS))

    if len(cd_candidates) == 0:
        return None

    # Either a median (all differing data points) or mode
    mode, freq = Counter(cd_candidates).most_common(1)[0]
    if freq > 1:
        return mode
    else:
        # Median
        return sorted(cd_candidates)[len(cd_candidates) // 2]


def estimate_g(velocities: list[tuple[int, Vector]], cd: float) -> float | None:
    # cd required, because vy = cd * vy - g
    if len(velocities) < 2:
        return None

    g_candidates = []
    for i in range(len(velocities) - 1):
        dt, v0 = velocities[i]
        _, v1 = velocities[i + 1]

        denom = 1 - cd**dt
        if abs(denom) < EPSILON_ANGLE:
            continue

        g_val = (cd**dt * v0.y - v1.y) * (1 - cd) / denom

        # Discard if it doesn't make sense.
        if g_val < 0:
            continue

        g_candidates.append(round(g_val, CDG_DIGITS))

    if len(g_candidates) == 0:
        return None

    # Either a median (all differing data points) or mode
    mode, freq = Counter(g_candidates).most_common(1)[0]
    if freq > 1:
        return mode
    else:
        # Median
        return sorted(g_candidates)[len(g_candidates) // 2]


def velocity_to_angles(vel: Vector) -> tuple[float, float]:
    horiz = math.hypot(vel.x, vel.z) + EPSILON_ANGLE
    pitch = math.degrees(math.atan2(vel.y, horiz))
    yaw = math.degrees(math.atan2(vel.x, vel.z))
    return yaw, pitch


def backpropagate(
    p: Vector, v: Vector, s: int, cd: float, g: float
) -> tuple[Vector, Vector]:
    p_back, v_back = Vector(p.x, p.y, p.z), Vector(v.x, v.y, v.z)
    for _ in range(s):
        prev_v = Vector(v_back.x, v_back.y + g, v_back.z).div(cd)
        prev_p = p_back.sub(prev_v)
        p_back, v_back = prev_p, prev_v

    return p_back, v_back


def avg_to_instantaneous_velocity(
    avg_vel: Vector, dt: int, cd: float, g: float
) -> Vector:
    """
    Convert an observed *average per-tick* velocity computed as:
        avg_vel = (p(t+dt) - p(t)) / dt
    into the instantaneous velocity v(t) at the start of that interval,
    under the discrete model:
        v_{k+1} = cd * v_k   (x,z)
        v_{k+1}.y = cd * v_k.y - g
    dt is integer number of ticks between observations.
    """
    if dt <= 0:
        return avg_vel.copy()

    # handle x,z (closed form)
    if abs(1 - cd) < EPSILON_ANGLE:
        # cd -> 1 (no drag) limit: displacement_x = v0.x * dt
        v_x = avg_vel.x
        v_z = avg_vel.z
    else:
        S = (1 - cd**dt) / (1 - cd)  # sum_{k=0..dt-1} cd^k
        v_x = avg_vel.x * dt / S
        v_z = avg_vel.z * dt / S

    # y component (closed form invert)
    if abs(1 - cd) < EPSILON_ANGLE:
        # cd -> 1 limit: v decreases linearly by g each tick
        # displacement_y = dt * v0.y - g * dt*(dt-1)/2
        # avg_y = v0.y - g*(dt-1)/2  => v0.y = avg_y + g*(dt-1)/2
        v_y = avg_vel.y + g * (dt - 1) / 2.0
    else:
        S = (1 - cd**dt) / (1 - cd)
        # A = dt/(1-cd) - (1 - cd^dt)/(1-cd)^2
        A = dt / (1 - cd) - (1 - cd**dt) / ((1 - cd) ** 2)
        # displacement_y = v0.y * S - g * A
        # avg_y = displacement_y / dt
        # => v0.y = (avg_y * dt + g * A) / S
        v_y = (avg_vel.y * dt + g * A) / S

    return Vector(v_x, v_y, v_z)


def compute_rmse(
    sim_traj: list[tuple[float, Vector]],
    obs_pts: list[tuple[float, Vector]],
    s: int,
    offsets: list[int],
) -> float:
    """
    Compute RMSE between simulated and observed trajectory segments.

    sim_traj: list of (tick, Vector) or plain list indexed by tick (you use simulate_trajectory which appends per tick).
    obs_pts: list of observed (time_seconds, Vector)
    s: number of ticks we backpropagated (muzzle is s ticks before obs_pts[0])
    offsets: integer tick offsets for each obs point relative to obs_pts[0], e.g.
             offsets[0] == 0, offsets[1] == ticks_between(obs1, obs0), ...
    """
    if len(offsets) == 0:
        return float("inf")

    last_offset = offsets[-1]
    if len(sim_traj) < s + last_offset + 1:
        return float("inf")

    err = 0.0
    n = len(obs_pts)
    for i in range(n):
        sim_idx = s + offsets[i]
        diff = sim_traj[sim_idx][1].sub(obs_pts[i][1])
        err += diff.dot(diff)
    return math.sqrt(err / n)


# TODO: clean everything up
@timed_function
def estimate_muzzle(
    partial_trajectory: list[tuple[float, Vector]],
    v_ms_range: tuple[int, int],
    c_d: float = None,
    g: float = None,
    max_s: int = 750,
):
    # 2 datapoints are enough if drag and gravity are already known.
    # 3 is enough, but weird things may still happen with c_d and g estimation.
    if len(partial_trajectory) < 2:
        return  # Not enough info

    obs_v = compute_obs_velocities(partial_trajectory)

    # TODO: if g is known, we can compute c_d more reliably
    c_d = estimate_cd(obs_v) if c_d is None else c_d
    if c_d is None:
        return  # Not enough info
    g = estimate_g(obs_v, c_d) if g is None else g
    if g is None:
        return  # Not enough info

    first_dt, first_avg_vel = obs_v[0]
    v_curr = avg_to_instantaneous_velocity(first_avg_vel, first_dt, c_d, g)

    # compute tick offsets for each observation relative to the first observed time
    t0 = partial_trajectory[0][0]
    offsets = [sec2tick(t - t0) for t, _ in partial_trajectory]

    best = dict(rmse=float("inf"), stats=None)
    for s in range(0, max_s):
        muzzle_pos, v0 = backpropagate(partial_trajectory[0][1], v_curr, s, c_d, g)
        v_ms = round(v0.length() * 20)  # From v/tick to v/sec

        # Outside these bounds, the velocity is absolutely off,
        # so we cut our losses and avoid the expensive simulation.
        # Additionally, cd and g are probably wrong too, so this makes
        # the worst case scenario take less long.
        if v_ms_range is not None:
            if v_ms < v_ms_range[0] or v_ms > v_ms_range[1]:
                continue

        yaw, pitch = velocity_to_angles(v0)
        # NOTE: length will mess with the simulation due to
        # yaw/pitch being AT v0, not actual muzzle location.
        cannon_candidate = Cannon(muzzle_pos, v_ms, 0, g, c_d, yaw, pitch, 0, 0)
        sim_ticks = s + offsets[-1]
        sim_traj = simulate_trajectory(cannon_candidate, max_ticks=sim_ticks)

        rmse = compute_rmse(sim_traj, partial_trajectory, s, offsets)
        if rmse < best["rmse"]:
            best["rmse"] = rmse
            best["stats"] = dict(
                pos=muzzle_pos,
                v_ms=v_ms,
                g=g,
                c_d=c_d,
                yaw=yaw,
                pitch=pitch,
            )

        # Good enough, we're satisfied.
        if best["rmse"] < ACCEPTABLE_RMSE:
            return best["stats"]

    return best["stats"]


def perform_simulation(
    cannon: Cannon,
    target_pos: Vector,
    fire_at_target: bool = True,
    trajectory_type: str = "low",
    perform_estimation: bool = False,
    radar: Radar = None,
    assumed_cd: float = None,
    assumed_g: float = None,
    assumed_v_ms_range: tuple[int, int] = None,
):
    results = dict()

    trajectory = []
    if not fire_at_target:
        trajectory = simulate_trajectory(cannon, stop_y=target_pos.y)
        results.update(dict(yaw=cannon.yaw, pitch=cannon.pitch))
    else:
        yaw, pitch, t = calculate_yaw_pitch_t(
            cannon, target_pos, trajectory_type == "low"
        )
        if yaw is not None and pitch is not None and t is not None:
            if pitch >= cannon.min_pitch and pitch <= cannon.max_pitch:
                cannon.yaw, cannon.pitch, max_ticks = yaw, pitch, round(t)
                trajectory = simulate_trajectory(cannon, max_ticks=max_ticks)

        results.update(dict(yaw=yaw, pitch=pitch))

    if len(trajectory) == 0:
        return results

    muzzle_pos = trajectory[0][1]
    tn, pos = trajectory[-1]
    results.update(
        dict(
            trajectory=trajectory,
            muzzle_pos=muzzle_pos,
            flight_time=tn,
            impact_pos=pos,
            error_impact=pos.sub(target_pos).length(),
        )
    )

    if not perform_estimation:
        return results

    observed_trajectory = get_observed_trajectory(trajectory, radar)
    results.update(dict(n_obs=len(observed_trajectory)))
    if len(observed_trajectory) == 0:
        return results

    results.update(dict(observed_trajectory=observed_trajectory))

    # TODO: this will go to inf if muzzle is PAST target! 
    stats = estimate_muzzle(
        partial_trajectory=observed_trajectory,
        v_ms_range=assumed_v_ms_range,
        c_d=assumed_cd,
        g=assumed_g,
    )
    if stats is None:
        return results

    est_muzzle_pos = stats["pos"]
    results.update(
        dict(
            est_muzzle_pos=est_muzzle_pos,
            est_v_ms=stats["v_ms"],
            est_g=stats["g"],
            est_c_d=stats["c_d"],
            est_yaw=stats["yaw"],
            est_pitch=stats["pitch"],
            error_est_muzzle_pos=muzzle_pos.sub(est_muzzle_pos).length(),
        )
    )
    return results
