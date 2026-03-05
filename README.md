# EWH Simulator (ewh_fleet_lib)

Electric Water Heater Fleet Simulation and Control library. This project provides a comprehensive suite of tools for modeling and simulating fleets of Electric Water Heaters (EWH), evaluating user water consumption profiles, and testing various power shedding and capping controls.

## Features

- **Fleet Simulation**: Create and simulate multiple electric water heaters with customizable device properties (volume, element power, dimensions).
- **Water Consumption Profiles**: Integrates stochastic usage models and real-world water draw data.
- **Control Strategies**: 
  - Standard thermostat hysteresis logic.
  - Fleet-wide power capping and reduction constraints.
  - Operator-triggered power shedding.
- **Visualizations**: Interactive plots and metric tracking powered by `plotly` and `streamlit`.

## Setup

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

Install the dependencies specified in `pyproject.toml` using `pip`:

```bash
pip install .
```

## Running the Simulator

You can explore the simulation logic natively by running the simulation environment script. For an interactive Streamlit dashboard:

```bash
streamlit run simulation_runner/simulation_env.py
```

This default script models the impact of power capping and shedding on a fleet, analyzing rebound behaviors and temperature impacts.

## Note on External Data
Due to file sizes, raw data (`*.csv`) and sample outputs (`*.png`) are intentionally excluded from version control. Please ensure you place the required data files in their respective folders when replicating environments.
