
from simulation_runner.simulation_env import EWHSimulator
from utils.data_handling import ewh_characteristics, ewh_water_consumption, ewh_power_consumption
from utils.plotting import PlotManager
import pandas as pd
import numpy as np
import copy
import streamlit as st
import plotly.graph_objects as go

n_heaters = 1000
duration_minutes = 24 * 60
shed_kw = 750
cap_kw = 100
time_window = (8 * 60, 9 * 60)
time_index = pd.date_range("2024-01-01", periods=duration_minutes, freq="1min")
data = {
    "characteristics": ewh_characteristics.head(n_heaters),
    "consumption": ewh_water_consumption,
    "parameters": {
        "max_temp_C": 58,
        "min_temp_C": 55,
        "initial_temp_C": 58,
        "control_strategy": "hysteresis",
        "hysteresis": 3,
    }
}

sim = EWHSimulator(mode="data", n_heaters=n_heaters, data=data)
sim.setup()
plotter = PlotManager()
capped_results = sim.simulate_power_capped(
        duration_minutes=duration_minutes,
        cap_kw=cap_kw,
        time_window=time_window,
        run_baseline=True,
        track_metrics=True
    )

# Extract results
temperatures = capped_results["temperatures"]
statuses = capped_results["statuses"]
metrics = capped_results["metrics"]

# Show impact metrics
window_start, window_end = time_window
pre_cap_power = metrics["pre_cap_power_kw"]
post_cap_power = metrics["total_power_kw"]

# Calculate power reduction
power_reduction = [pre - post for pre, post in zip(pre_cap_power, post_cap_power)]

# Create a time index for the simulation
time_index = pd.date_range("2024-01-01", periods=len(pre_cap_power), freq="1min")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=time_index,
    y=capped_results["baseline_power"],
    mode="lines",
    name="Baseline Power",
    line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=time_index,
    y=capped_results["capped_power"],
    mode="lines",
    name="Capped Power",
    line=dict(color="red")
))
fig.add_shape(
    type="rect",
    x0=time_index[window_start],
    x1=time_index[window_end - 1],
    y0=0,
    y1=max(capped_results["baseline_power"]) * 1.1,
    fillcolor="gray",
    opacity=0.2,
    layer="below",
    line_width=0
)
fig.update_layout(
    title=f"Impact of {cap_kw}kW Power Cap",
    xaxis_title="Time",
    yaxis_title="Aggregate Power (kW)"
)
st.plotly_chart(fig, use_container_width=True)

# Show temperature impact
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=time_index,
    y=metrics["avg_temp_all"],
    mode="lines",
    name="Avg Temp - All Heaters"
))
fig2.add_trace(go.Scatter(
    x=[time_index[i] for i in range(len(metrics["avg_temp_affected"])) if
       metrics["avg_temp_affected"][i] is not None],
    y=[t for t in metrics["avg_temp_affected"] if t is not None],
    mode="lines",
    name="Avg Temp - Affected Heaters"
))
fig2.add_shape(
    type="rect",
    x0=time_index[window_start],
    x1=time_index[window_end - 1],
    y0=min([t for t in metrics["avg_temp_all"] if t is not None]) * 0.95,
    y1=max([t for t in metrics["avg_temp_all"] if t is not None]) * 1.05,
    fillcolor="gray",
    opacity=0.2,
    layer="below",
    line_width=0
)
fig2.update_layout(
    title="Temperature Impact of Power Capping",
    xaxis_title="Time",
    yaxis_title="Temperature (°C)"
)
st.plotly_chart(fig2, use_container_width=True)

if metrics["pre_cap_power_kw"] and metrics["total_power_kw"]:
    # Calculate actual power reduction at each timestep
    power_reduction = [pre - post for pre, post in
                       zip(metrics["pre_cap_power_kw"], metrics["total_power_kw"])]

    # Plot power reduction over time
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=time_index,
        y=power_reduction,
        mode="lines",
        name="Power Reduction",
        line=dict(color="orange")
    ))
    # Highlight capping window
    fig3.add_shape(
        type="rect",
        x0=time_index[window_start],
        x1=time_index[window_end - 1],
        y0=0,
        y1=max(power_reduction) * 1.1,
        fillcolor="gray",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    fig3.update_layout(
        title="Power Reduction Over Time",
        xaxis_title="Time",
        yaxis_title="Power Reduction (kW)"
    )
    st.plotly_chart(fig3, use_container_width=True)
    if metrics["pre_cap_power_kw"] and metrics["total_power_kw"]:
        # Create a new figure showing relationship between cap and actual power
        fig_cap = go.Figure()

        # Add trace for pre-cap power (baseline)
        fig_cap.add_trace(go.Scatter(
            x=time_index,
            y=metrics["pre_cap_power_kw"],
            mode="lines",
            name="Pre-cap Power",
            line=dict(color="blue")
        ))

        # Add trace for post-cap power
        fig_cap.add_trace(go.Scatter(
            x=time_index,
            y=metrics["total_power_kw"],
            mode="lines",
            name="Post-cap Power",
            line=dict(color="red")
        ))

        # Add horizontal line showing the cap
        fig_cap.add_shape(
            type="line",
            x0=time_index[window_start],
            x1=time_index[window_end - 1],
            y0=cap_kw,
            y1=cap_kw,
            line=dict(color="green", width=2, dash="dash"),
        )

        # Highlight capping window
        fig_cap.add_shape(
            type="rect",
            x0=time_index[window_start],
            x1=time_index[window_end - 1],
            y0=0,
            y1=max(metrics["pre_cap_power_kw"]) * 1.1,
            fillcolor="gray",
            opacity=0.2,
            layer="below",
            line_width=0
        )

        fig_cap.update_layout(
            title="Power Capping Verification",
            xaxis_title="Time",
            yaxis_title="Power (kW)"
        )

        st.plotly_chart(fig_cap, use_container_width=True)