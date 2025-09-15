import math
import random
from dataclasses import dataclass
from time import time
from lib.utils import *

EPSILON = 1e-12


@dataclass
class Cannon:
    pos: Vector
    v_ms: float
    length: int
    g: float
    cd: float
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


def solve_pitch(
    distance: float,
    velocity_ms: float,
    target_height: float,
    cannon_length: int,
    gravity: float,
    drag_coefficient: float,
    low: bool,
):
    """
    All math comes from Endal's ballistics calculator made in Desmos (https://www.desmos.com/calculator/az4angyumw),
    there may be bugs because the formulas sure look like some kind of alien language to me. It's >60x faster than
    brute-forcing pitch, and has higher precision to boot.
    """

    # Constants
    g = gravity  # CBC gravity
    cd = drag_coefficient  # drag coefficient
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
    A = g * cd / (u * (1 - cd))

    def B(t):
        return t * (g * cd / (1 - cd)) * 1 / X_R

    C = L / (u * X_R) * (g * cd / (1 - cd)) + h / X_R

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
            p1_X_R1 = u * math.cos(a_R1) / math.log(cd)
            p2_X_R1 = cd**t - 1
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
    # The target is literally inside the barrel
    if dpos.length() <= cannon.length:
        return None, None, None
    horizontal_dist = math.sqrt(dpos.x**2 + dpos.z**2)
    t, pitch = solve_pitch(
        distance=horizontal_dist,
        velocity_ms=cannon.v_ms,
        target_height=dpos.y,
        cannon_length=cannon.length,
        gravity=cannon.g,
        drag_coefficient=cannon.cd,
        low=low,
    )

    if t is None:
        return None, None, None

    yaw = -math.degrees(math.atan2(dpos.x, dpos.z))
    yaw = (yaw + 360) % 360  # -180, 180 => 0, 360
    return yaw, pitch, t


def simulate_trajectory(
    cannon: Cannon, max_ticks: int = None, stop_y: float = None
) -> list[tuple[int, Vector]]:
    """
    According to @sashafiesta#1978's formula on Discord:
    Vx = 0.99 * Vx
    Vy = 0.99 * Vy - 0.05
    """
    assert (max_ticks != None) or (stop_y != None), "At least one of these must be set."

    yaw_rad = math.radians(cannon.yaw)
    pitch_rad = math.radians(cannon.pitch)

    # Following CBC convention:
    # 0/360 => +Z
    # 90 => -X
    # 180 => -Z
    # 370 => +X
    # Cannon aiming direction
    dir = Vector(
        math.cos(pitch_rad) * -math.sin(yaw_rad),
        math.sin(pitch_rad),
        math.cos(pitch_rad) * math.cos(yaw_rad),
    )
    # Muzzle (initial projectile) position
    pos = cannon.pos.add(dir.mul(cannon.length))
    # v/s -> v/t
    vel = dir.mul(cannon.v_ms / 20)

    trajectory: list[tuple[int, Vector]] = []

    if max_ticks is None and stop_y is not None:
        max_ticks = 100000  # Endless calculation protection.

    for t in range(max_ticks + 1):
        trajectory.append((t, pos.copy()))
        pos = pos.add(vel)
        vel = vel.mul(cannon.cd)
        vel.y -= cannon.g
        # We'll never reach, so quit.
        if stop_y is not None and pos.y < stop_y and vel.y <= 0:
            break

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


def calculate_velocities(
    observations: list[tuple[float, Vector]],
) -> list[tuple[int, Vector]]:
    """
    Note that this will return list of len(datapoints) - 1, because we can't know
    the velocity of the first observation. Also, dt is in ticks, not seconds.
    """
    velocities = []
    for i in range(len(observations) - 1):
        t0, p0 = observations[i]
        t1, p1 = observations[i + 1]
        dt = sec2tick(t1 - t0)  # You never know what in-game lag will do.
        if dt <= 0:
            continue
        vel = p1.sub(p0).div(dt)
        velocities.append((dt, vel))
    return velocities


def velocity_to_angles(vel: Vector) -> tuple[float, float]:
    horiz = math.hypot(vel.x, vel.z) + EPSILON
    pitch = math.degrees(math.atan2(vel.y, horiz))
    yaw = -math.degrees(math.atan2(vel.x, vel.z))
    yaw = (yaw + 360) % 360  # -180, 180 => 0, 360
    return yaw, pitch


def estimate_cd(velocities: list[int, Vector], g: float = None) -> float | None:
    if len(velocities) < 2:
        return

    def calc_cdxz(v0c: float, v1c: float, dt: int) -> float | None:
        if abs(v0c) > EPSILON and v1c * v0c > EPSILON:
            return (v1c / v0c) ** (1 / dt)

    def solve_cdy(
        v0y: float,
        v1y: float,
        dt: int,
        g: float,
    ) -> float | None:
        # No information to be gained, don't bother.
        if dt <= 0 or (abs(v0y) < EPSILON and abs(v1y) < EPSILON):
            return

        # Difference between predicted and observed vy
        def f(cd: float) -> float:
            # Clamp domain to avoid weird shit.
            if cd <= 0 or cd >= 1:
                return float("inf")
            cd_dt = cd**dt
            return cd_dt * v0y - g * (1 - cd_dt) / (1 - cd) - v1y

        # Bisection
        low, high = EPSILON, 1 - EPSILON
        f_low, f_high = f(low), f(high)
        if f_low * f_high > 0:
            return  # no guaranteed root

        while high - low > EPSILON:
            middle = (low + high) / 2
            f_mid = f(middle)
            if f_low * f_mid <= 0:
                high, f_high = middle, f_mid
            else:
                low, f_low = middle, f_mid

        return (low + high) / 2

    low_angle_threshold = 10
    cd_candidates = []
    for i in range(len(velocities) - 1):
        dt, v0 = velocities[i]
        _, v1 = velocities[i + 1]
        if g is None:  # Default case where we don't know shit.
            cdx = calc_cdxz(v0.x, v1.x, dt)
            cdz = calc_cdxz(v0.z, v1.z, dt)
            if cdx is not None:
                cd_candidates.append(cdx)
            if cdz is not None:
                cd_candidates.append(cdz)
        else:
            cdy = solve_cdy(v0.y, v1.y, dt, g)
            cdy is not None and cd_candidates.append(cdy)
            # For flat angles, cdy becomes less reliable
            # while cdxz becomes a bit better.
            _, pitch = velocity_to_angles(v0)
            if abs(pitch) <= low_angle_threshold:
                cdx = calc_cdxz(v0.x, v1.x, dt)
                cdz = calc_cdxz(v0.z, v1.z, dt)
                if cdx is not None:
                    cd_candidates.append(cdx)
                if cdz is not None:
                    cd_candidates.append(cdz)

    if len(cd_candidates) == 0:
        return

    best = sorted(cd_candidates)[len(cd_candidates) // 2]  # Median
    return round_increment(best, EPSILON)


def estimate_g(velocities: list[tuple[int, Vector]], cd: float) -> float | None:
    # cd required, because vy = cd * vy - g
    if len(velocities) < 2:
        return

    g_candidates = []
    for i in range(len(velocities) - 1):
        dt, v0 = velocities[i]
        _, v1 = velocities[i + 1]

        denom = 1 - cd**dt
        if abs(denom) < EPSILON:
            continue

        # Dont discard negative g. It will skew the median.
        g_val = (cd**dt * v0.y - v1.y) * (1 - cd) / denom
        g_candidates.append(g_val)

    if len(g_candidates) == 0:
        return

    # Median
    best = sorted(g_candidates)[len(g_candidates) // 2]
    return round_increment(best, EPSILON)


def avg_to_inst_velocity(dt: int, vel: Vector, cd: float, g: float) -> Vector:
    """
    Average velocity (dt) converted to velocity at the start of the tick.
    """
    result = vel.copy()
    if dt <= 0:
        return result

    # drag factor
    fd = 1 - cd
    if abs(fd) >= EPSILON:  # cd < 1 means there's drag.
        cd_dt = cd**dt
        S = (1 - cd_dt) / fd
        result.x *= dt / S
        result.z *= dt / S
        A = (dt * fd - (1 - cd_dt)) / (fd**2)
        result.y = (result.y * dt + g * A) / S
        return result
    else:
        result.y += g * (dt - 1) / 2
        return result


def reverse_step(pos: Vector, vel: Vector, cd: float, g: float):
    # A single step of simulate trajectory, but in reverse.
    vel.y += g
    vel = vel.div(cd)
    pos = pos.sub(vel)
    return pos, vel


def compute_rmse(
    sim_traj: list[tuple[float, Vector]],
    observations: list[tuple[float, Vector]],
    t: int,
    offsets: list[int],
) -> float:
    """
    See how well the forward simulated points match up with the
    observed points by calculating the root mean square error.
    """
    last_offset = offsets[-1]
    if len(sim_traj) < t + last_offset + 1:
        return float("inf")

    total_sq_error = 0
    for i in range(len(observations)):
        sim_idx = t + offsets[i]
        dpos = sim_traj[sim_idx][1].sub(observations[i][1])
        total_sq_error += dpos.dot(dpos)

    return math.sqrt(total_sq_error / len(observations))


def estimate_muzzle(
    observations: list[tuple[float, Vector]],
    min_vms: int = None,
    max_vms: int = None,
    vms_multiple: int = None,
    cd: float = None,
    g: float = None,
    max_t: int = 750,
):
    # 2 datapoints is enough only if drag and gravity are already known.
    if len(observations) < 2:
        return

    min_vms = min_vms if min_vms is not None else 0
    max_vms = max_vms if max_vms is not None else 10**9  # Supposed to be infinite.

    velocities = calculate_velocities(observations)
    # If cd or g are unknown, we use a fallback system to infer the values.
    # Note: the fallback can fail.
    cd = cd if cd is not None else estimate_cd(velocities, g)
    g = g if g is not None else estimate_g(velocities, cd)
    if cd is None or g is None:
        return None

    earliest_vel = avg_to_inst_velocity(velocities[0][0], velocities[0][1], cd, g)
    # tick offsets for each observation relative to the first observed time
    offsets = [sec2tick(t - observations[0][0]) for t, _ in observations]

    best_rmse = float("inf")
    acceptable_threshold = 0.001
    pos, vel = observations[0][1], earliest_vel
    for t in range(0, max_t):
        vms = round(vel.length() * 20)  # From v/tick to v/sec

        # Only simulate forward if velocity is plausible.
        if vms > max_vms:
            break  # Will only grow in value as t increases. Give up.

        if min_vms <= vms and (vms_multiple is None or vms % vms_multiple == 0):
            yaw, pitch = velocity_to_angles(vel)
            # Remember: at muzzle, so length is 0.
            sim_traj = simulate_trajectory(
                Cannon(pos, vms, 0, g, cd, yaw, pitch, 0, 0),
                max_ticks=t + offsets[-1],
            )
            best_rmse = min(best_rmse, compute_rmse(sim_traj, observations, t, offsets))
            if best_rmse < acceptable_threshold:
                return dict(cd=cd, g=g, pitch=pitch, pos=pos, t=t, v_ms=vms, yaw=yaw)

        # Continue simulating backward.
        pos, vel = reverse_step(pos, vel, cd, g)
    return


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
    assumed_v_ms_multiple: int = None,
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

    # I don't want to touch the frontend anymore 💀
    minvms, maxvms = (None, None) if assumed_v_ms_range is None else assumed_v_ms_range
    stats = estimate_muzzle(
        observations=observed_trajectory,
        min_vms=minvms,
        max_vms=maxvms,
        vms_multiple=assumed_v_ms_multiple,
        cd=assumed_cd,
        g=assumed_g,
    )
    if stats is None:
        return results

    est_muzzle_pos = stats["pos"]
    results.update(
        dict(
            error_est_muzzle_pos=muzzle_pos.sub(est_muzzle_pos).length(),
            est_cd=stats["cd"],
            est_g=stats["g"],
            est_muzzle_pos=est_muzzle_pos,
            est_pitch=stats["pitch"],
            est_t=stats["t"],
            est_v_ms=stats["v_ms"],
            est_yaw=stats["yaw"],
        )
    )
    return results
