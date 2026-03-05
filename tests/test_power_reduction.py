# import pytest
import numpy as np
from simulation_runner.simulation_env import EWHSimulator
from utils.data_handling import ewh_characteristics, ewh_water_consumption, ewh_power_consumption
from utils.plotting import PlotManager
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# Define simulation parameters
n_heaters = 1000
duration_minutes = 24 * 60
reduction_kw = 850
time_window = (8 * 60, 9 * 60)  # 8:00-9:00 AM

# Set up simulation with real data
data = {
    "characteristics": ewh_characteristics.head(n_heaters),
    "consumption": ewh_water_consumption,
    "parameters": {
        "max_temp_C": 58,
        "min_temp_C": 55,
        "initial_temp_C": 58,
        "control_strategy": "hysteresis",
        "hysteresis": 3,
        "n_days": 3,  # Only simulate for one day
    }
}

# Run simulation
sim = EWHSimulator(mode="data", n_heaters=n_heaters, data=data)
sim.setup()
results = sim.simulate_power_reduction(
    duration_minutes=duration_minutes,
    reduction_kw=reduction_kw,
    time_window=time_window,
    run_baseline=True,
    track_metrics=True
)

# Create time index for plotting
time_index = pd.date_range("2024-01-01", periods=duration_minutes, freq="1min")

# Create power comparison plot
fig = go.Figure()

# Add baseline power trace
fig.add_trace(go.Scatter(
    x=time_index,
    y=results["baseline_power"],
    mode="lines",
    name="Baseline Power",
    line=dict(color="blue")
))

# Add reduced power trace
fig.add_trace(go.Scatter(
    x=time_index,
    y=results["reduced_power"],
    mode="lines",
    name="Reduced Power",
    line=dict(color="red", dash="dash")
))

# Add shaded region for reduction window
window_start, window_end = time_window
fig.add_shape(
    type="rect",
    x0=time_index[window_start],
    x1=time_index[window_end],
    y0=0,
    y1=max(results["baseline_power"]) * 1.05,
    fillcolor="gray",
    opacity=0.2,
    layer="below",
    line_width=0
)

# Update layout
fig.update_layout(
    title="Power Reduction Comparison",
    xaxis_title="Time",
    yaxis_title="Power (kW)",
    template="plotly_white"
)

# Display the figure interactively
fig.show()




# def mpc_control(fleet, prediction_horizon=24 * 60, time_window=(8 * 60, 9 * 60), power_cap=500):
#     """Control fleet using MPC with power cap during specified window"""
#
#     # 1. Gather current state
#     current_temps = [h.current_temp_C for h in fleet.heaters]
#
#     # 2. Predict water consumption for each heater
#     predicted_water_draws = predict_water_consumption(prediction_horizon)
#
#     # 3. Set up optimization problem
#     from pyomo.environ import ConcreteModel, Var, Binary, Constraint, Objective, minimize, SolverFactory
#
#     model = ConcreteModel()
#     # Decision variables: heating element status for each heater at each timestep
#     model.status = Var(range(len(fleet.heaters)), range(prediction_horizon), domain=Binary)
#
#     # Temperature evolution constraints
#     # (would model how temperature changes based on heating decisions)
#
#     # Power constraint during cap window
#     def power_cap_rule(model, t):
#         if time_window[0] <= t < time_window[1]:
#             return sum(model.status[i, t] * fleet.heaters[i].heating_element_power_w
#                        for i in range(len(fleet.heaters))) <= power_cap * 1000
#         else:
#             return Constraint.Skip
#
#     model.power_cap = Constraint(range(prediction_horizon), rule=power_cap_rule)
#
#     # Comfort constraints (temperature bounds)
#
#     # Objective: minimize cost while maximizing comfort
#     model.objective = Objective(expr=objective_function(model), sense=minimize)
#
#     # 4. Solve optimization
#     solver = SolverFactory('gurobi')  # or another solver
#     results = solver.solve(model)
#
#     # 5. Apply first-step decisions to fleet
#     for i, heater in enumerate(fleet.heaters):
#         heater.set_heating_element_status(model.status[i, 0].value > 0.5)
#
#     return model.objective()