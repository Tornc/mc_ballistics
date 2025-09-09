import numpy as np
import streamlit as st
import plotly.graph_objects as go

from lib.utils import Vector
from lib.simulation import (
    Cannon,
    Radar,
    simulate_trajectory,
    get_observed_trajectory,
    estimate_muzzle,
)


def add_trace(
    fig: go.Figure,
    positions: Vector | list[Vector],
    mode: str = "markers",
    name: str | None = None,
    size: int | None = None,
):
    # Convert single vector to list to make life easier.
    if isinstance(positions, Vector):
        positions = [positions]

    # NOTE: we swap y and z on purpose. Because Y should be UP according to
    # Minecraft conventions.
    x = [p.x for p in positions]
    y = [p.z for p in positions]
    z = [p.y for p in positions]

    trace_config = dict(x=x, y=y, z=z, mode=mode, name=name, text=[name])

    if "markers" in mode:
        trace_config["marker"] = dict(size=size)
    if "lines" in mode:
        trace_config["line"] = dict(width=size)

    fig.add_trace(go.Scatter3d(**trace_config))


def add_hemisphere(fig: go.Figure, pos: Vector, radius: float, name: str | None = None):
    u = np.linspace(0, 2 * np.pi, 25)  # azimuth angle
    v = np.linspace(0, np.pi / 2, 13)  # polar angle (0-90 degrees)

    # On purpose
    x = pos.x + radius * np.outer(np.cos(u), np.sin(v))
    y = pos.z + radius * np.outer(np.sin(u), np.sin(v))
    z = pos.y + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.15,
            showscale=False,
            name=name,
            hoverinfo="skip",
            showlegend=True,
        )
    )


def ax_style(rng: tuple[int]):
    return dict(
        showbackground=False,
        zeroline=True,
        showgrid=True,
        zerolinewidth=4,
        zerolinecolor="grey",
        gridcolor="LightGrey",
        minallowed=rng[0],
        maxallowed=rng[1],
    )


def init_plot(area: dict, width: int = None, height: int = None):
    fig = go.Figure()
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            # intentional axis swapping
            xaxis=ax_style(area["x"]),
            yaxis=ax_style(area["z"]),
            zaxis=ax_style(area["y"]),
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Z",
            zaxis_title="Y",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=-0.05, y=-0.05, z=-0.1),
                eye=dict(x=-0.9, y=-1.45, z=1.3),
            ),
        ),
        legend=dict(x=0, y=0, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def generate_plot(
    fig: go.Figure,
    cannon: Cannon,
    radar: Radar,
    target_pos: Vector,
    max_ticks: int,
):
    trajectory = simulate_trajectory(cannon, max_ticks)
    add_trace(fig, [p for _, p in trajectory], "lines", "Trajectory", 8)
    observed_trajectory = get_observed_trajectory(trajectory, radar)
    # TODO: fix weird edge case that bypasses this check somehow
    if len(observed_trajectory) > 0:
        print(len(observed_trajectory))
        add_trace(fig, [p for _, p in observed_trajectory], "markers", "Observation", 4)
        est_muzzle_pos, info = estimate_muzzle(observed_trajectory)
        if est_muzzle_pos is not None:
            print(f"Est: { est_muzzle_pos}")
            add_trace(fig, est_muzzle_pos, "markers+text", "Muzzle (estimate)")

    add_trace(fig, cannon.pos, "markers+text", "Cannon", 12)
    add_trace(fig, target_pos, "markers+text", "Target", 12)
    add_hemisphere(fig, radar.pos, radar.range, "Radar range")

    return fig
