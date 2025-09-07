def _finite_diffs(points_np: np.ndarray):
    N = points_np.shape[0]
    v = np.zeros_like(points_np)
    if N >= 3:
        v[1:-1] = (points_np[2:] - points_np[:-2]) / 2.0
        v[0] = points_np[1] - points_np[0]
        v[-1] = points_np[-1] - points_np[-2]
    elif N == 2:
        v[0] = points_np[1] - points_np[0]
        v[1] = v[0].copy()
    else:
        v[0] = np.zeros(3)
    return v

def _simulate_from_state(p0: np.ndarray, v0: np.ndarray, cd: float, g: float, max_ticks: int):
    pts = np.zeros((max_ticks, 3), dtype=float)
    pts[0] = p0.copy()
    vx, vy, vz = v0.copy()
    # note: we store p[0] = p0; then for t in 1.. we update velocity and position
    for t in range(1, max_ticks):
        vx *= cd
        vy = vy * cd - g
        vz *= cd
        pts[t] = pts[t-1] + np.array([vx, vy, vz])
    return pts

def _interp_sim_at(sim_pts: np.ndarray, t_float: float):
    """
    Linear interpolate simulated positions at fractional tick t_float.
    If t_float < 0 -> return first sample; if > last -> return last sample.
    """
    if t_float <= 0.0:
        return sim_pts[0].copy()
    t_max = sim_pts.shape[0] - 1
    if t_float >= t_max:
        return sim_pts[-1].copy()
    t0 = int(math.floor(t_float))
    a = t_float - t0
    p0 = sim_pts[t0]
    p1 = sim_pts[t0 + 1]
    return (1.0 - a) * p0 + a * p1

def locate_muzzle(observed_points: list[tuple[float,float,float]],
                  cd_prior: float = 0.99,
                  g_prior: float = 0.05,
                  max_sim_ticks: int = 750,
                  tau_margin_lower: int = 20):
    """
    Returns (muzzle_vector, diagnostics)
    muzzle_vector has attributes .x,.y,.z
    diagnostics contains: optimized v0_per_tick, tau, residual_norm, success, message
    """
    obs = np.asarray(observed_points, dtype=float)
    N = obs.shape[0]
    if N < 2:
        raise ValueError("Need at least two observed points.")

    # ---- initial guesses ----
    v_est = _finite_diffs(obs)
    mid_idx = N // 2
    v_mid = v_est[mid_idx].copy()   # per-tick velocity at observed middle sample

    # back-propagate a conservative number of ticks to guess muzzle position
    # compute a robust estimate of ticks back via distance / speed_per_tick
    speed_per_tick = max(1e-6, np.linalg.norm(v_mid))
    # distance from mid observed to origin-of-coordinates isn't great; instead use distance from mid observed to first observed
    # better heuristic: assume observed mid is somewhere along flight; approximate ticks back = distance(mid->first)/speed_per_tick + small
    # simpler: estimate ticks from mid to muzzle by assuming muzzle lies further back than observed:
    est_ticks_back = int(max(30, min(500, round(np.linalg.norm(obs[mid_idx] - obs[0]) / speed_per_tick * 1.5 + 40))))
    est_ticks_back = max(25, min(est_ticks_back, max_sim_ticks//2))

    # conservative lower bound on tau: don't allow tau to be extremely small (this fixes the "not far back enough" problem).
    tau_init = est_ticks_back
    tau_lb = max(0.0, est_ticks_back - tau_margin_lower)
    tau_ub = max_sim_ticks - N - 1
    if tau_ub < tau_lb:
        tau_ub = max(tau_lb, tau_lb + 50)

    # inverse dynamics roll-back to get p0 guess and v0 guess at muzzle (per-tick)
    p_guess = obs[mid_idx].copy()
    vtmp = v_mid.copy()
    for _ in range(est_ticks_back):
        vtmp = np.array([vtmp[0] / cd_prior, (vtmp[1] + g_prior) / cd_prior, vtmp[2] / cd_prior])
        p_guess = p_guess - vtmp

    # state vector: [p0x,p0y,p0z, vx,vy,vz, tau]
    x0 = np.concatenate([p_guess, vtmp, np.array([float(tau_init)])])

    # bounds
    rng = 1280
    obs_min = obs.min(axis=0) - rng
    obs_max = obs.max(axis=0) + rng
    lb = np.concatenate([obs_min, np.array([-rng, -rng / 2, -rng]), np.array([tau_lb])])
    ub = np.concatenate([obs_max, np.array([rng, rng, rng]), np.array([tau_ub])])

    # residual function uses fractional tau and interpolation (no sliding integer alignment)
    def residual_fn(x):
        p0 = x[0:3]
        v0 = x[3:6]
        tau = float(x[6])
        # simulate (must be long enough)
        sim = _simulate_from_state(p0, v0, cd_prior, g_prior, max_sim_ticks)
        res = np.zeros((N, 3), dtype=float)
        # compute residual at times tau + i
        for i in range(N):
            t_query = tau + float(i)
            sim_pos = _interp_sim_at(sim, t_query)
            res[i] = sim_pos - obs[i]
        # flatten
        return res.ravel()

    # run optimizer
    res = least_squares(residual_fn, x0, bounds=(lb, ub), verbose=2, max_nfev=2000, xtol=1e-8, ftol=1e-8)

    p0_opt = res.x[0:3]
    v0_opt = res.x[3:6]
    tau_opt = float(res.x[6])

    # final residual norm
    final_sim = _simulate_from_state(p0_opt, v0_opt, cd_prior, g_prior, max_sim_ticks)
    residuals = []
    for i in range(N):
        residuals.append(_interp_sim_at(final_sim, tau_opt + i) - obs[i])
    residuals = np.array(residuals)
    resid_norm = float((residuals.ravel()**2).sum())

    diagnostics = {
        "muzzle": Vector(*p0_opt),
        "v0_per_tick": v0_opt.tolist(),
        "tau_ticks": tau_opt,
        "residual_norm": resid_norm,
        "success": bool(res.success),
        "message": res.message,
        "initial_est_ticks_back": est_ticks_back,
    }
    return diagnostics["muzzle"], diagnostics













# ------------------


def estimate_cd_and_g(obs_vels: list[Vector]) -> tuple[float, float]:
    """
    Single-pass estimator, because radar is overpowered and has no noise.
    If issues arise, use 2-pass median estimator instead.
    """

    # This means we need 3 total trajectory observations, as velocity
    # observations is n-1 in length.
    if len(obs_vels) < 2:
        return None, None

    v0 = obs_vels[0]
    v1 = obs_vels[1]

    if abs(v0.x) > 1e-6:
        cd = v1.x / v0.x
    elif abs(v0.z) > 1e-6:
        cd = v1.z / v0.z
    else:
        return None, None
    g = cd * v0.y - v1.y
    return cd, g





EPSILON_ANGLE = 1e-12
EARLY_EXIT_RMSE = 1e-3
DEFAULT_CD = 0.99
DEFAULT_G = 0.05


def compute_obs_velocities(points: list[Vector]) -> list[Vector]:
    velocities = []
    for i in range(len(points) - 1):
        velocities.append(points[i + 1].sub(points[i]))
    return velocities


def velocity_to_angles(vel: Vector) -> tuple[float, float]:
    horiz = math.hypot(vel.x, vel.z) + EPSILON_ANGLE
    pitch = math.degrees(math.atan2(vel.y, horiz))
    yaw = math.degrees(math.atan2(vel.x, vel.z))
    return yaw, pitch


def estimate_cd_and_g(obs_vels: list[Vector]) -> tuple[float, float]:
    # This means we need 3 total trajectory observations, as velocity
    # observations is n-1 in length.
    if len(obs_vels) < 2:
        return None, None
    
    cd_ratios = []
    g_estimates = []

    for i in range(len(obs_vels) - 1):
        v0 = obs_vels[i]
        v1 = obs_vels[i + 1]

        # Horizontal ratios for cd
        if abs(v0.x) > 1e-6:
            cd_ratios.append(v1.x / v0.x)
        if abs(v0.z) > 1e-6:
            cd_ratios.append(v1.z / v0.z)

    # Use median ratio as cd estimate
    cd_hint = sorted(cd_ratios)[len(cd_ratios) // 2] if cd_ratios else 0.99

    # Second pass (depends on cd_hint)
    for i in range(len(obs_vels) - 1):
        v0 = obs_vels[i]
        v1 = obs_vels[i + 1]
        g_estimates.append(cd_hint * v0.y - v1.y)

    g_hint = sorted(g_estimates)[len(g_estimates) // 2] if g_estimates else 0.05

    return cd_hint, g_hint


def backpropagate(
    p: Vector, v: Vector, s: int, cd: float, g: float
) -> tuple[Vector, Vector]:
    p_back, v_back = Vector(p.x, p.y, p.z), Vector(v.x, v.y, v.z)
    for _ in range(s):
        prev_v = Vector(v_back.x, v_back.y + g, v_back.z).div(cd)
        prev_p = p_back.sub(prev_v)
        p_back, v_back = prev_p, prev_v

    return p_back, v_back


def compute_rmse(sim_traj: list[Vector], obs_pts: list[Vector], s: int) -> float:
    """
    Compute RMSE between simulated and observed trajectory segments.
    """
    if len(sim_traj) < s + len(obs_pts):
        return float("inf")
    err = 0.0
    len_obs = len(obs_pts)
    for i in range(len_obs):
        diff = sim_traj[s + i].sub(obs_pts[i])
        err += diff.dot(diff)
    return math.sqrt(err / len_obs)


# TODO: cleanup and understand if it's iffy or not.
# also understand and clean up the magic numbers
# TODO: what if observations are not in 1 tick interval?
# TODO: adapt to new format.
@timed_function
def locate_muzzle(
    partial_trajectory: list[tuple[float, Vector]],
    max_s: int = 750,
    cd_grid: tuple[float] = (0.985, 0.99),
    g_grid: tuple[float] = (0.04, 0.05),
    speed_candidates: list[int] = list(range(40, 341, 20)),
):
    if len(partial_trajectory) < 2:
        return None, None  # not enough info

    pts = partial_trajectory
    print(pts)
    obs_v = compute_obs_velocities(pts)
    # use the first observed per-tick velocity as current velocity at t_obs0
    v_curr = obs_v[0]

    best = {"rmse": float("inf"), "muzzle": None, "params": None}

    # cd_hint, g_hint = estimate_cd_and_g(obs_v)
    # if cd_hint is None or g_hint is None:
    #     cd_hint, g_hint = DEFAULT_CD, DEFAULT_G

    # # Expand CD/G grids to include the estimated values
    # cd_list = sorted(set(cd_grid + (cd_hint,)))
    # g_list = sorted(set(g_grid + (g_hint,)))

    # len_obs = len(pts)

    # for cd in cd_list:
    #     for g in g_list:
    #         # try a reasonable small range of s first (closer detection is more likely)
    #         for s in range(0, min(max_s, 501)):
    #             muzzle_pos, v0 = backpropagate(partial_trajectory[0], v_curr, s, cd, g)
    #             speed_ms = v0.length() * 20  # From v/tick to v/sec
    #             # round to nearest allowed speed candidate
    #             best_speed = min(speed_candidates, key=lambda s_c: abs(s_c - speed_ms))
    #             yaw, pitch = velocity_to_angles(v0)
    #             # simulate with length=0
    #             cannon_candidate = Cannon(muzzle_pos, best_speed, 0, g, cd)

    #             sim_ticks = s + len_obs + 5  # a bit of margin
    #             sim_traj = simulate_trajectory(cannon_candidate, yaw, pitch, sim_ticks)
    #             rmse = compute_rmse(sim_traj, pts, s)

    #             if rmse < best["rmse"]:
    #                 best["rmse"] = rmse
    #                 best["muzzle"] = muzzle_pos
    #                 best["params"] = {
    #                     "s": s,
    #                     "cd": cd,
    #                     "g": g,
    #                     "speed_ms": best_speed,
    #                     "yaw": yaw,
    #                     "pitch": pitch,
    #                     "v0_per_tick": v0,
    #                 }

    #             # early exit heuristic: if rmse is tiny, return immediately
    #             if best["rmse"] < EARLY_EXIT_RMSE:
    #                 return best["muzzle"], best

    return best["muzzle"], best