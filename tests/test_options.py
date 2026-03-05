import pytest
from simulation_runner.simulation_env import EWHSimulator
from utils.data_handling import ewh_characteristics, ewh_water_consumption, ewh_power_consumption
import pandas as pd
import numpy as np
from utils.plotting import PlotManager
from scipy.stats import norm


def test_normal_operation(n_heaters=2):
    # Use the first 2 heaters for a quick test
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
    fleet = sim.fleet
    results = sim.simulate_all(duration_minutes=24*60)

    # Extract data for plotting
    n_heaters = len(results)
    timesteps = len(results[0]['statuses'])
    timestep_s = 60  # Assuming 1-min timestep
    time_index = pd.date_range("2024-01-01", periods=timesteps, freq="1min")
    temperatures = np.array([results[i]['temperatures'] for i in range(n_heaters)])
    statuses = np.array([results[i]['statuses'] for i in range(n_heaters)])
    power_profiles = np.array([np.array(results[i]['statuses'][:-1]) * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0 for i in range(n_heaters)])
    agg_power = power_profiles.sum(axis=0)
    df = data['characteristics']

    plotter = PlotManager()
    plotter.plot_power_profiles(time_index, power_profiles, agg_power, ewh_power_consumption=ewh_power_consumption)
    plotter.plot_boxplot(df)


def test_data_mode_with_real_data(n_heaters=2):
    # Use the first 2 heaters for a quick test
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
    fleet = sim.fleet
    # results = sim.run(duration_minutes=24*60)
    duration_minutes = 24 * 60
    max_total_power_kw = 0  # Example cap
    shut_down_power_kw = 100  # Example shutdown power
    time_window = (8*60, 9*60 )  # Example: apply cap from minute 60 to 600
    results = []

    n_heaters = len(fleet.heaters)
    temperatures = [[] for _ in range(n_heaters)]
    statuses = [[] for _ in range(n_heaters)]

    for t in range(duration_minutes):
        # 1. Let each heater decide ON/OFF by its own logic
        for heater in fleet.heaters:
            heater.control_heating_element()
        # 2. Apply power shedding if in the window
        if time_window[0] <= t < time_window[1]:
            fleet.apply_power_shedding(max_total_power_kw)
            # fleet.apply_power_shutdown(shut_down_power_kw)
        # 3. Update state, skipping control logic
        for i, heater in enumerate(fleet.heaters):
            temp, status = heater.step(t, skip_control=True)
            temperatures[i].append(temp)
            statuses[i].append(int(status))

    # for t in range(duration_minutes):
    #     for heater in fleet.heaters:
    #         heater.control_heating_element()
    #     if time_window[0] <= t < time_window[1]:
    #         total_on_power = sum(h.heating_element_power_w for h in fleet.heaters if h.heating_element_status) / 1000.0
    #         print(f"Before shutdown at t={t}: {total_on_power} kW ON")
    #         fleet.apply_power_shutdown(shut_down_power_kw)
    #         total_on_power_after = sum(
    #             h.heating_element_power_w for h in fleet.heaters if h.heating_element_status) / 1000.0
    #         print(f"After shutdown at t={t}: {total_on_power_after} kW ON")
    #     for i, heater in enumerate(fleet.heaters):
    #         temp, status = heater.step(t, skip_control=True)
    #         temperatures[i].append(temp)
    #         statuses[i].append(int(status))
    import copy
    for t in range(duration_minutes):
        # 1. Simulate what would happen WITHOUT shedding
        fleet_copy = copy.deepcopy(fleet)
        for heater in fleet_copy.heaters:
            heater.control_heating_element()
        uncontrolled_on_power = sum(
            h.heating_element_power_w for h in fleet_copy.heaters if h.heating_element_status
        ) / 1000.0

        if time_window[0] <= t < time_window[1]:
            print(f"Uncontrolled power at t={t}: {uncontrolled_on_power} kW ON")
            # 2. Apply shedding to the real fleet based on the uncontrolled state
            # Use the uncontrolled ON/OFF status to decide which heaters to turn OFF
            future_statuses = [h.heating_element_status for h in fleet_copy.heaters]
            fleet.apply_power_shutdown_given_statuses(shut_down_power_kw, future_statuses)
            total_on_power_after = sum(
                h.heating_element_power_w for h in fleet.heaters if h.heating_element_status
            ) / 1000.0
            print(f"After shutdown at t={t}: {total_on_power_after} kW ON")
        else:
            # Normal control if not in window
            for heater in fleet.heaters:
                heater.control_heating_element()

        # 3. Advance the real fleet (skip control, since already applied)
        for i, heater in enumerate(fleet.heaters):
            temp, status = heater.step(t, skip_control=True)
            temperatures[i].append(temp)
            statuses[i].append(int(status))

    temperatures = np.array(temperatures)
    statuses = np.array(statuses)
    timestep_s = 60

    time_index = pd.date_range("2024-01-01", periods=duration_minutes, freq="1min")
    power_profiles = np.array([
        statuses[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
        for i in range(n_heaters)
    ])
    agg_power = power_profiles.sum(axis=0)
    df = data['characteristics']
    plotter = PlotManager()
    plotter.plot_power_profiles(time_index, power_profiles, agg_power, ewh_power_consumption=ewh_power_consumption)
    plotter.plot_boxplot(df)

def test_data_mode_with_real_data_baseline_shedding(n_heaters=2):
    import copy
    # Prepare data and simulator
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
    fleet = sim.fleet
    duration_minutes = 24 * 60
    shut_down_power_kw = 100  # Example shutdown power
    time_window = (8 * 60, 9 * 60)  # Example: apply shedding from minute 480 to 540

    # 1. Simulate baseline (no shedding)
    baseline_temperatures = [[] for _ in range(n_heaters)]
    # baseline_statuses = [[] for _ in range(n_heaters)]
    # Deepcopy fleet for baseline to avoid state contamination
    # 1. Simulate baseline (no shedding)
    baseline_fleet = copy.deepcopy(fleet)
    baseline_statuses = [[] for _ in range(n_heaters)]
    for t in range(duration_minutes):
        for heater in baseline_fleet.heaters:
            heater.control_heating_element()
        for i, heater in enumerate(baseline_fleet.heaters):
            _, status = heater.step(t, skip_control=True)
            baseline_statuses[i].append(int(status))
    baseline_statuses = np.array(baseline_statuses)

    # 2. Simulate shedded scenario (shed relative to baseline)
    shedded_fleet = copy.deepcopy(fleet)
    shedded_statuses = [[] for _ in range(n_heaters)]
    for t in range(duration_minutes):
        # Get baseline ON/OFF for this timestep
        baseline_on = [baseline_statuses[i][t] for i in range(n_heaters)]
        # Normal control
        for heater in shedded_fleet.heaters:
            heater.control_heating_element()
        # Apply shedding in window
        if time_window[0] <= t < time_window[1]:
            # Only consider heaters that would be ON in baseline
            shedded_fleet.apply_power_shutdown_given_statuses(shut_down_power_kw, baseline_on)
        # Advance state
        for i, heater in enumerate(shedded_fleet.heaters):
            _, status = heater.step(t, skip_control=True)
            shedded_statuses[i].append(int(status))
    shedded_power_profiles = np.array([
        shedded_statuses[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
        for i in range(n_heaters)
    ])

    baseline_statuses = np.array(baseline_statuses)
    shedded_statuses = np.array(shedded_statuses)

    # Calculate baseline power profiles and aggregate power
    baseline_power_profiles = np.array([
        baseline_statuses[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
        for i in range(n_heaters)
    ])
    agg_power_baseline = baseline_power_profiles.sum(axis=0)

    shedded_power_profiles = np.array([
        shedded_statuses[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
        for i in range(n_heaters)
    ])
    agg_power_shedded = shedded_power_profiles.sum(axis=0)

    # 3. Plot results
    time_index = pd.date_range("2024-01-01", periods=duration_minutes, freq="1min")
    plotter = PlotManager()
    plotter.plot_aggregate_power_comparison(
        time_index, agg_power_baseline, agg_power_shedded,
        label_baseline="Baseline (no shedding)",
        label_shedding="With real shedding (rebound visible)"
    )




    # Extract data for plotting
    # n_heaters = len(results)
    # timesteps = len(results[0]['statuses'])
    # timestep_s = 60  # Assuming 1-min timestep
    # time_index = pd.date_range("2024-01-01", periods=timesteps, freq="1min")
    # # temperatures = np.array([results[0]['temperatures']])  # Example: first heater
    # temperatures = np.array([results[i]['temperatures'] for i in range(n_heaters)])
    # statuses = np.array([results[i]['statuses'] for i in range(n_heaters)])
    # power_profiles = np.array([np.array(results[i]['statuses'][:-1]) * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0 for i in range(n_heaters)])
    # agg_power = power_profiles.sum(axis=0)
    # df = data['characteristics']
    #
    #
    # plotter = PlotManager()
    # # plotter.plot_temperature(time_index, temperatures)
    # # plotter.plot_status(time_index, statuses)
    # plotter.plot_power_profiles(time_index, power_profiles, agg_power, ewh_power_consumption=ewh_power_consumption)
    # plotter.plot_boxplot(df)
test_normal_operation(n_heaters=1000)
# test_data_mode_with_real_data(n_heaters=1000)
# test_data_mode_with_real_data_baseline_shedding(n_heaters=1000)