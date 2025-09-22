import math
import random
from dataclasses import dataclass
from time import time
from lib.utils import *

EPSILON = 1e-9

# === General ===


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
class Target:
    pos: Vector
    vel: Vector


@dataclass
class Radar:
    pos: Vector
    range: int
    scan_rate: int
    drop_rate: float


def perform_simulation(
    cannon: Cannon,
    target: Target,
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
        trajectory = simulate_trajectory(cannon, stop_y=target.pos.y)
        results.update(dict(yaw=cannon.yaw, pitch=cannon.pitch))
    else:
        # Hideous code btw.
        if target.vel.length() == 0:
            rs = calculate_solution(cannon, target.pos, trajectory_type == "low")
            if rs is not None:
                yaw, pitch, t = rs
                if pitch >= cannon.min_pitch and pitch <= cannon.max_pitch:
                    cannon.yaw, cannon.pitch, max_ticks = yaw, pitch, round(t)
                    trajectory = simulate_trajectory(cannon, max_ticks=max_ticks)

                results.update(dict(yaw=yaw, pitch=pitch))
        else:
            rs = calculate_solution_moving(
                cannon, target.pos, target.vel, trajectory_type == "low"
            )
            if rs is not None:
                yaw, pitch, t = rs
                path = simulate_target(target, round(t))
                target.pos = path[-1][1]
                results.update(target_path=path)
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
            error_impact=pos.sub(target.pos).length(),
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


# === Simulation ===


def simulate_target(target: Target, ticks: int) -> list[tuple[int, Vector]]:
    path = []
    pos = target.pos
    for t in range(ticks + 1):
        path.append((t, pos.copy()))
        pos = pos.add(target.vel)
    return path


def simulate_trajectory(
    cannon: Cannon, max_ticks: int = None, stop_y: float = None
) -> list[tuple[int, Vector]]:
    """
    `max_ticks` and/or `stop_y` must be provided.

    According to @sashafiesta#1978's formulas on Discord:
    - Vx = 0.99 * Vx
    - Vy = 0.99 * Vy - 0.05

    Args:
        cannon (Cannon):
        max_ticks (int, optional): Defaults to None.
        stop_y (float, optional): Defaults to None.

    Returns:
        list[tuple[int, Vector]]: Projectile trajectory; timestamps + positions.
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
    vel = dir.mul(cannon.v_ms / 20)  # v/s -> v/t

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
    A semi-realistic depiction of how what an MC radar would have to deal with.
    1. First observation happens at an arbitrary time.
    2. Timestamps are in seconds.
    3. Radar range is a thing.
    4. Radar scan rate may be slow for balancing reasons.
    5. Lag exists and/or a scan is computationally expensive.

    Args:
        trajectory (list[tuple[int, Vector]]): timestamp in **ticks**, position
        radar (Radar):

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


# === Ballistic Calculator ===


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
                return t, math.degrees(a_R1)

            # We've passed the target (aka we're close), now oscillate around the actual
            # target value until it's 'good enough' or it's taking too long.
            if (increasing_t and X_R1 > X_R) or (not increasing_t and X_R1 < X_R):
                increasing_t = not increasing_t
                break

            t += step_size if increasing_t == low else -step_size

        # Increase the precision after breaking out, since we're closer to target
        step_size = step_size / 2

    return t, math.degrees(a_R1)


def calculate_solution(
    cannon: Cannon, target_pos: Vector, low: bool
) -> tuple[float, float, float] | None:
    """
    Calculates the angles and flight time for either the low or high
    pitch solution given a target coordinate.

    Args:
        cannon (Cannon):
        target_pos (Vector):
        low (bool):

    Returns:
        tuple[float, float, float] | None: (yaw, pitch, flight time)
    """
    dpos = target_pos.sub(cannon.pos)
    # The target is literally inside the barrel
    if dpos.length() <= cannon.length:
        return

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
        return

    # Following CBC convention.
    yaw = -math.degrees(math.atan2(dpos.x, dpos.z))
    yaw = (yaw + 360) % 360  # -180, 180 => 0, 360
    return yaw, pitch, t


def calculate_solution_moving(
    cannon: Cannon, target_pos: Vector, target_vel: Vector, low: bool
) -> tuple[float, float, float] | None:
    """
    Calculates the angles and flight time for either the low or high
    pitch solution given a target position AND target velocity.

    Args:
        cannon (Cannon):
        target_pos (Vector):
        target_vel (Vector):
        low (bool):

    Returns:
        tuple[float, float, float] | None: (yaw, pitch, flight time)
    """
    threshold = 0.5  # well, dt is an int and we're rounding anyway.
    num_halvings = 12
    step_size = 75

    t0, tn = 0, 750
    t = t0 if low else tn
    increasing_t = low
    for _ in range(num_halvings):
        while True:
            # TODO: this is a lazy (slow) way to do it. Ideally, you'd create an
            # entirely new solver so you avoid this double solving loop. But I'm
            # kinda lazy so that's a problem for in the future.
            pred_pos = target_pos.add(target_vel.mul(t))
            result = calculate_solution(cannon, pred_pos, low)
            if result is not None:
                _, _, shell_t = result
                error = shell_t - t

                if abs(error) <= threshold:
                    return result

                if (increasing_t and error < 0) or (not increasing_t and error > 0):
                    increasing_t = not increasing_t
                    t += step_size if increasing_t == low else -step_size
                    break

            t += step_size if increasing_t == low else -step_size
            if t0 > t or t > tn:  # We've searched the entire space; give up.
                return None

        step_size /= 2

    return None


# === Reverse-location ===


def calculate_velocities(obs: list[tuple[int, Vector]]) -> list[tuple[int, Vector]]:
    """
    Args:
        obs (list[tuple[int, Vector]]): observations with normalised timestamps.

    Returns:
        list[tuple[int, Vector]]: (dt, velocity XYZ)
    """
    velocities = []
    for i in range(len(obs) - 1):
        t0, p0 = obs[i]
        t1, p1 = obs[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        vel = p1.sub(p0).div(dt)
        velocities.append((dt, vel))
    return velocities


def estimate_cd(velocities: list[int, Vector], g: float = None) -> float | None:
    if len(velocities) < 2:
        return

    def calc_cdxz(v0c: float, v1c: float, dt: int) -> float | None:
        if abs(v0c) > EPSILON and v1c * v0c > EPSILON:
            return (v1c / v0c) ** (1 / dt)

    def solve_cdy(v0y: float, v1y: float, dt: int, g: float) -> float | None:
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
            mid = (low + high) / 2
            f_mid = f(mid)
            if f_low * f_mid <= 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

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


def step_backward(pos: Vector, vel: Vector, cd: float, g: float):
    """
    A single step of simulate trajectory, but in reverse.
    """
    p, v = pos.copy(), vel.copy()
    v.y += g
    v = v.div(cd)
    p = p.sub(v)
    return p, v


def step_forward(pos: Vector, vel: Vector, cd: float, g: float):
    """
    A single step of simulate_trajectory()
    """
    p = pos.add(vel)
    v = vel.mul(cd)
    v.y -= g
    return p, v


def forward_state(
    p0: Vector, v0: Vector, cd: float, g: float, t: int
) -> tuple[Vector, Vector]:
    """
    Closed form of simulate_trajectory()

    Args:
        p0 (Vector):
        v0 (Vector):
        cd (float):
        g (float):
        t (int): how many ticks to jump forward.

    Returns:
        tuple[Vector, Vector]: pN, vN
    """
    # No drag
    if abs(1 - cd) < EPSILON:
        vt = v0.copy()
        vt.y -= g * t
        av = v0.mul(t)
        av.y -= g * t * (t - 1) / 2
        pt = p0.add(av)
        return pt, vt

    cd_t = cd**t
    S = (1 - cd_t) / (1 - cd)
    # Velocity
    vt = v0.mul(cd_t)
    vt.y -= g * S
    # Position
    av = v0.mul(S)
    av.y += (g / (1 - cd)) * (S - t)
    pt = p0.add(av)
    return pt, vt


def evaluate(
    obs: list[tuple[int, Vector]],
    p0: Vector,
    v0: Vector,
    cd: float,
    g: float,
    t0: int,
    threshold: float = 1e-3,  # Honestly idk what this should be.
) -> bool:
    """
    Args:
        obs (list[tuple[int, Vector]]): Our real-world samples that we need to check with.
        p0 (Vector): Proposed muzzle position.
        v0 (Vector): Proposed muzzle velocity.
        cd (float):
        g (float):
        t0 (int): Proposed time of firing the cannon; how far in the past it is. (<= 0)
        threshold (float, optional): Max distance any simulated position can be from an observed one. Defaults to 1e-3.

    Returns:
        bool: Whether we accept or reject pos and vel.
    """
    pos, vel = p0.copy(), v0.copy()
    prev_t = t0
    for t, p in obs:
        dt = t - prev_t
        prev_t = t
        if dt == 1:
            # Individually, cheaper than solving.
            pos, vel = step_forward(pos, vel, cd, g)
        else:
            pos, vel = forward_state(pos, vel, cd, g, dt)
        # We know that the correct muzzle position (p0) will
        # give an exact fit, so we return early to reduce wasted
        # simulation. This is also good for worst case (no fit ever).
        if pos.sub(p).length() > threshold:
            return False

    return True


def velocity_to_angles(vel: Vector) -> tuple[float, float]:
    """
    Calculates the direction of motion based on the velocity vector.
    Args:
        vel (Vector):

    Returns:
        tuple[float, float]: yaw, pitch
    """
    horiz = math.hypot(vel.x, vel.z) + EPSILON
    pitch = math.degrees(math.atan2(vel.y, horiz))
    yaw = -math.degrees(math.atan2(vel.x, vel.z))
    yaw = (yaw + 360) % 360  # -180, 180 => 0, 360
    return yaw, pitch


def estimate_muzzle(
    observations: list[tuple[float, Vector]],
    min_vms: int = None,
    max_vms: int = None,
    vms_multiple: int = None,
    cd: float = None,
    g: float = None,
    max_t: int = 750,
) -> dict | None:
    """
    Notes:
        Be careful with your assumptions, conservative estimates are better than getting nothing at all.

    Args:
        observations (list[tuple[float, Vector]]): Samples with format: (seconds, Vector(X, Y, Z)).
        min_vms (int, optional): Rule out everything below. Defaults to None.
        max_vms (int, optional): Rule out everything above. Defaults to None.
        vms_multiple (int, optional): Rule out everything that's not a multiple (m/s). Defaults to None.
        cd (float, optional): drag coefficient (0-1). Defaults to None.
        g (float, optional): gravity (0-1). Defaults to None.
        max_t (int, optional): How many ticks we can look back in time. Defaults to 750.

    Returns:
        dict | None: Will return nothing if no candidate is fit.
    """
    # 2 is enough only if drag and gravity are already known.
    if len(observations) < 2:
        return

    # m/s -> m/t
    min_v: float = min_vms / 20 if min_vms is not None else 0
    max_v: float = max_vms / 20 if max_vms is not None else 10**9  # Treat as infinite.
    v_multiple: float = vms_multiple / 20 if vms_multiple is not None else None

    # Normalise the timestamps by making them relative to the first observation.
    t0 = observations[0][0]
    obs = [(sec2tick(t - t0), p) for t, p in observations]
    vel_obs = calculate_velocities(obs)

    # If cd or g are unknown, we use a fallback system to infer the values.
    # This requires 3 observations instead of 2.
    # NOTE: fallback is more likely to fail with flatter trajectories.
    cd = cd if cd is not None else estimate_cd(vel_obs, g)
    g = g if g is not None else estimate_g(vel_obs, cd)
    if cd is None or g is None:
        return

    # pos, vel here are the proposed muzzle position and velocity we need to verify.
    pos = obs[0][1]
    vel = avg_to_inst_velocity(vel_obs[0][0], vel_obs[0][1], cd, g)
    for t in range(0, -max_t - 1, -1):
        # Only step back on the 2nd iteration. GOTO would've been nicer. (Lua-pilled)
        if t < 0:
            pos, vel = step_backward(pos, vel, cd, g)

        # See reasoning below guard clauses.
        v_mag = vel.length()
        v_mag_rounded = round_increment(v_mag, 0.05)

        # Only bother evaluating if the velocity is plausible.
        # NOTE: Velocity will only increase the further back in time we go.
        if v_mag_rounded > max_v:
            break  # Only gets worse. Give up.
        if v_mag_rounded < min_v:
            continue  # We may end up in bounds later, so continue.
        if v_multiple and v_mag_rounded % v_multiple != 0:
            continue

        # We scale it so the velocity magnitude will be a whole number in m/s.
        # NOTE: Rounding the velocity **MUST** be done, otherwise you'll get garbage.
        vel_rounded = vel.mul(v_mag / v_mag_rounded)
        if evaluate(obs, pos, vel_rounded, cd, g, t):
            # Include some extra information that may come in handy besides the muzzle position.
            yaw, pitch = velocity_to_angles(vel_rounded)
            # How many ticks in the past (relative to first observation) the cannon has fired.
            t_obs0 = -t
            vms = int(v_mag_rounded * 20)  # m/t -> m/s
            return dict(cd=cd, g=g, pitch=pitch, pos=pos, t=t_obs0, v_ms=vms, yaw=yaw)

    return
