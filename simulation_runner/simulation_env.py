import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from ewh_model.single_ewh import ElectricWaterHeater
from ewh_model.ewh_fleet import ElectricWaterHeaterFleet
import plotly.graph_objects as go
import streamlit as st
import copy
import numpy as np

class EWHSimulator:
    def __init__(self, mode="random", n_heaters=10, data=None, user_config=None):
        self.mode = mode
        self.n_heaters = n_heaters
        self.data = data
        self.user_config = user_config
        self.fleet = None

    def setup(self):
        if self.mode == "random":
            self.fleet = ElectricWaterHeaterFleet(self.n_heaters)
            self.fleet.create_randomized_heaters(self.n_heaters)
        elif self.mode == "data" and self.data is not None:
            heaters = []
            for i, row in self.data["characteristics"].head(self.n_heaters).iterrows():
                demand_profile = self.data["consumption"][[f'vdot_{i}_lps']].copy()
                demand_profile = demand_profile.rename(columns={f'vdot_{i}_lps': 'total'})
                heater = ElectricWaterHeater(
                    n_users=None,
                    volume_liters=row['volume_liters'],
                    diameter_m=row['diameter_m'],
                    height_m=row['height_m'],
                    heating_element_power_w=row['heating_element_power_w'],
                    demand_profile=demand_profile,
                    **self.data.get("parameters", {})
                )
                heaters.append(heater)
            self.fleet = ElectricWaterHeaterFleet(n_heaters=len(heaters))
            self.fleet.heaters = heaters
        elif self.mode == "user" and self.user_config is not None:
            heaters = [
                ElectricWaterHeater(**cfg)
                for cfg in self.user_config[:self.n_heaters]
            ]
            self.fleet = ElectricWaterHeaterFleet(n_heaters=len(heaters))
            self.fleet.heaters = heaters

    def run(self, duration_minutes=24*60, power_shedding_kw=None, time_window=None):
        if self.fleet is None:
            self.setup()
        if power_shedding_kw is not None and time_window is not None:
            return self.fleet.simulate_all_with_power_shedding(duration_minutes, power_shedding_kw, time_window)
        else:
            return self.fleet.simulate_all(duration_minutes)

    def simulate_baseline(self, duration_minutes=24 * 60):
        if self.fleet is None:
            self.setup()
        n_heaters = len(self.fleet.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]
        for t in range(duration_minutes):
            for i, heater in enumerate(self.fleet.heaters):

                temp, status = heater.step(t, skip_control=False)
                temperatures[i].append(temp)
                statuses[i].append(int(status))
        return np.array(temperatures), np.array(statuses)

    def simulate_all(self, duration_minutes):
        results = []
        for ewh in self.fleet.heaters:
            temps, statuses = ewh.simulate_heating(duration_minutes)
            results.append({
                "temperatures": temps,
                "statuses": statuses,
                "demand_profile": ewh.demand_profile
            })
        return results

    # def simulate_power_capped(self, duration_minutes=24 * 60, cap_kw=100, time_window=(8 * 60, 9 * 60)):
    #     if self.fleet is None:
    #         self.setup()
    #     n_heaters = len(self.fleet.heaters)
    #     temperatures = [[] for _ in range(n_heaters)]
    #     statuses = [[] for _ in range(n_heaters)]
    #
    #     for heater in self.fleet.heaters:
    #         heater.set_heating_element_status(False)
    #         heater.control_heating_element()
    #
    #     for t in range(duration_minutes):
    #         # 1. Normal control
    #         for heater in self.fleet.heaters:
    #             heater.control_heating_element()
    #         # 2. Apply power cap in the window
    #         if time_window[0] <= t < time_window[1]:
    #             self.fleet.apply_power_shedding(cap_kw)
    #         # 3. Advance state
    #         for i, heater in enumerate(self.fleet.heaters):
    #             temp, status = heater.step(t, skip_control=True)
    #             temperatures[i].append(temp)
    #             statuses[i].append(int(status))
    #     return np.array(temperatures), np.array(statuses)
    def simulate_power_capped(self, duration_minutes=24 * 60, cap_kw=100, time_window=(8 * 60, 9 * 60),
                              run_baseline=True, track_metrics=True):
        """
        Simulate fleet operation with power capping during specified time window.

        Parameters:
            duration_minutes: Total simulation duration in minutes
            cap_kw: Maximum allowed power in kW during capping period
            time_window: Tuple of (start_minute, end_minute) for capping
            run_baseline: Whether to also run an uncapped baseline for comparison
            track_metrics: Track detailed metrics about capping impact

        Returns:
            dict: Simulation results containing temperatures, statuses, and metrics
        """
        import copy
        import numpy as np

        # Run baseline simulation if requested
        baseline_agg_power = None
        if run_baseline:
            baseline_sim = copy.deepcopy(self)
            _, baseline_statuses = baseline_sim.simulate_baseline(duration_minutes)
            baseline_power_profiles = np.array([
                baseline_statuses[i] * self.fleet.heaters[i].heating_element_power_w / 1000.0
                for i in range(len(self.fleet.heaters))
            ])
            baseline_agg_power = baseline_power_profiles.sum(axis=0)

        # Initialize simulation
        if self.fleet is None:
            self.setup()
        n_heaters = len(self.fleet.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]

        # Track metrics
        metrics = {
            "total_power_kw": [],
            "pre_cap_power_kw": [],
            "shed_counts": [],
            "avg_temp_all": [],
            "avg_temp_affected": []
        } if track_metrics else None

        # Initial control pass to set starting status
        for heater in self.fleet.heaters:
            heater.set_heating_element_status(False)
            heater.control_heating_element()

        # Main simulation loop
        for t in range(duration_minutes):
            pre_cap_power = baseline_agg_power[t]

            # Normal control on actual fleet
            for heater in self.fleet.heaters:
                heater.control_heating_element()

            # 2. Apply power cap in the window
            if time_window[0] <= t < time_window[1]:
                result = self.fleet.apply_power_shedding(cap_kw)

            if track_metrics:
                metrics["pre_cap_power_kw"].append(pre_cap_power)
            # Calculate total power after capping
            post_cap_power = sum(h.heating_element_power_w / 1000.0 for h in self.fleet.heaters
                                 if h.heating_element_status)
            if track_metrics:
                metrics["total_power_kw"].append(post_cap_power)

            # 4. Advance state
            current_temps = []
            for i, heater in enumerate(self.fleet.heaters):
                temp, status = heater.step(t, skip_control=True)
                temperatures[i].append(temp)
                statuses[i].append(int(status))
                current_temps.append(temp)

            # Update temperature metrics
            if track_metrics:
                metrics["avg_temp_all"].append(np.mean(current_temps))

        # Calculate additional metrics
        if track_metrics:
            window_start, window_end = time_window
            metrics["energy_saved_kwh"] = np.sum(baseline_agg_power[window_start:window_end] -
                                                 np.array(metrics["total_power_kw"][window_start:window_end])) / 60

        return {
            "temperatures": np.array(temperatures),
            "statuses": np.array(statuses),
            "baseline_power": baseline_agg_power,
            "capped_power": metrics["total_power_kw"] if track_metrics else None,
            "metrics": metrics
        }

    def simulate_power_reduction(self, duration_minutes=24 * 60, reduction_kw=700, time_window=(8 * 60, 9 * 60),
                                 run_baseline=True, track_metrics=True):
        # Run baseline simulation first to get pre-reduction power values
        baseline_sim = copy.deepcopy(self)
        _, baseline_statuses = baseline_sim.simulate_baseline(duration_minutes)
        baseline_power_profiles = np.array([
            baseline_statuses[i] * self.fleet.heaters[i].heating_element_power_w / 1000.0
            for i in range(len(self.fleet.heaters))
        ])
        baseline_agg_power = baseline_power_profiles.sum(axis=0)

        # Initialize simulation
        if self.fleet is None:
            self.setup()
        n_heaters = len(self.fleet.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]

        # Track metrics
        metrics = {
            "total_power_kw": [],
            "pre_reduction_power_kw": [],
            "shed_counts": [],
            "avg_temp_all": [],
            "avg_temp_affected": []
        } if track_metrics else None

        # Main simulation loop
        for t in range(duration_minutes):
            # Get baseline power for this timestep
            pre_reduction_power = baseline_agg_power[t] if run_baseline else None

            # Normal control on actual fleet
            for heater in self.fleet.heaters:
                heater.control_heating_element()

            # Apply power reduction in the window
            if time_window[0] <= t < time_window[1]:
                # Calculate target power directly from baseline
                # Target = baseline at this timestep minus reduction
                target_power = max(baseline_agg_power[t] - reduction_kw, 0)

                # Turn off heaters until we reach target power
                # Sort ON heaters by temperature descending (highest temp first)
                on_heaters = [(i, h) for i, h in enumerate(self.fleet.heaters) if h.heating_element_status]
                on_heaters.sort(key=lambda x: -x[1].current_temp_C)

                # Turn off all heaters first, then turn back on up to target
                for _, heater in on_heaters:
                    heater.set_heating_element_status(False)

                # Calculate total available power if all selected heaters were ON
                available_power = sum(h.heating_element_power_w / 1000.0 for _, h in on_heaters)

                # Turn heaters back on until we reach target (coldest first)
                on_heaters.sort(key=lambda x: x[1].current_temp_C)  # Sort by ascending temperature
                current_power = 0

                for _, heater in on_heaters:
                    heater_power = heater.heating_element_power_w / 1000.0
                    if current_power + heater_power <= target_power:
                        heater.set_heating_element_status(True)
                        current_power += heater_power
                    else:
                        # We've reached our target, keep the rest off
                        break

            # Track metrics
            if track_metrics:
                metrics["pre_reduction_power_kw"].append(pre_reduction_power)

                # Calculate total power after reduction
                post_reduction_power = sum(h.heating_element_power_w / 1000.0
                                           for h in self.fleet.heaters if h.heating_element_status)
                metrics["total_power_kw"].append(post_reduction_power)

            # Advance state
            current_temps = []
            for i, heater in enumerate(self.fleet.heaters):
                temp, status = heater.step(t, skip_control=True)
                temperatures[i].append(temp)
                statuses[i].append(int(status))
                current_temps.append(temp)

            # Update temperature metrics
            if track_metrics:
                metrics["avg_temp_all"].append(np.mean(current_temps))

        # Calculate additional metrics
        if track_metrics:
            window_start, window_end = time_window
            metrics["energy_saved_kwh"] = np.sum(baseline_agg_power[window_start:window_end] -
                                                 np.array(metrics["total_power_kw"][window_start:window_end])) / 60

        return {
            "temperatures": np.array(temperatures),
            "statuses": np.array(statuses),
            "baseline_power": baseline_agg_power,
            "reduced_power": metrics["total_power_kw"] if track_metrics else None,
            "metrics": metrics
        }

    def simulate_operator_shedding(self, duration_minutes=24 * 60, shed_kw=200, time_window=(8 * 60, 9 * 60)):
        if self.fleet is None:
            self.setup()
        n_heaters = len(self.fleet.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]
        forced_off = set()
        for heater in self.fleet.heaters:
            heater.set_heating_element_status(False)
            heater.control_heating_element()
        for t in range(duration_minutes):
            # 1. Normal control
            for heater in self.fleet.heaters:
                heater.control_heating_element()
            # 2. At the start of the window, select heaters to shed
            if t == time_window[0]:
                on_heaters = [
                    (i, h.heating_element_power_w)
                    for i, h in enumerate(self.fleet.heaters)
                    if h.heating_element_status
                ]
                # Sort by power descending (or any other strategy)
                on_heaters.sort(key=lambda x: -x[1])
                total = 0
                forced_off = set()
                for i, power in on_heaters:
                    forced_off.add(i)
                    total += power / 1000.0
                    if total >= shed_kw:
                        break
            # 3. During the window, force selected heaters OFF
            if time_window[0] <= t < time_window[1]:
                for i in forced_off:
                    self.fleet.heaters[i].set_heating_element_status(False)
            # 4. Advance state
            for i, heater in enumerate(self.fleet.heaters):
                temp, status = heater.step(t, skip_control=True)
                temperatures[i].append(temp)
                statuses[i].append(int(status))
            # 5. After window, clear forced_off
            if t == time_window[1]:
                forced_off = set()
        return np.array(temperatures), np.array(statuses)

    def simulate_operator_shedding_alternative(self, duration_minutes=24 * 60, shed_kw=700,
                                               time_window=(8 * 60, 9 * 60)):
        import copy
        if self.fleet is None:
            self.setup()
        n_heaters = len(self.fleet.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]
        baseline_at_window_start = None

        for t in range(duration_minutes):
            # 1. Simulate what would happen WITHOUT shedding (hysteresis only)
            fleet_copy = copy.deepcopy(self.fleet)
            for heater in fleet_copy.heaters:
                heater.control_heating_element()
            baseline_statuses = [h.heating_element_status for h in fleet_copy.heaters]
            baseline_on_power = sum(
                h.heating_element_power_w for h in fleet_copy.heaters if h.heating_element_status
            ) / 1000.0

            if t == time_window[0]:
                baseline_at_window_start = baseline_on_power

            # 2. Apply shedding if in window
            if time_window[0] <= t < time_window[1]:
                if baseline_at_window_start is not None:
                    target_power = max(baseline_on_power - shed_kw, 0)
                    print(f"t={t}, baseline_on_power={baseline_on_power}, target_power={target_power}")
                else:
                    target_power = 0

                # Find which heaters to turn ON to get as close as possible to target_power
                on_indices = [i for i, s in enumerate(baseline_statuses) if s]
                # Sort ON heaters by power descending (greedy)
                on_indices_sorted = sorted(on_indices, key=lambda i: -self.fleet.heaters[i].heating_element_power_w)
                current_power = 0
                allowed_on = []
                for i in on_indices_sorted:
                    p = self.fleet.heaters[i].heating_element_power_w / 1000.0
                    if current_power + p <= target_power + 1e-6:  # small tolerance
                        allowed_on.append(i)
                        current_power += p
                # Apply: only these heaters ON, others OFF
                for i, heater in enumerate(self.fleet.heaters):
                    if i in allowed_on:
                        heater.set_heating_element_status(True)
                    else:
                        heater.set_heating_element_status(False)
            else:
                # Normal control outside window
                for i, heater in enumerate(self.fleet.heaters):
                    heater.set_heating_element_status(baseline_statuses[i])

            # 3. Advance state (skip control, as statuses are set)
            for i, heater in enumerate(self.fleet.heaters):
                temp, status = heater.step(t, skip_control=True)
                temperatures[i].append(temp)
                statuses[i].append(int(status))

        return np.array(temperatures), np.array(statuses)

    def simulate_all_with_power_shedding(self, duration_minutes, power_shedding_kw, time_window):
        n_heaters = len(self.heaters)
        temperatures = [[] for _ in range(n_heaters)]
        statuses = [[] for _ in range(n_heaters)]
        for t in range(duration_minutes):
            # 1. Let each heater decide ON/OFF by its own logic
            for h in self.heaters:
                h.control_heating_element()
            # 2. Apply power shedding if in window
            if time_window[0] <= t < time_window[1]:
                self.apply_power_shedding(power_shedding_kw)
            # 3. Update state (do not call control logic again in step)
            for i, h in enumerate(self.heaters):
                temp, status = h.step(t, skip_control=True)
                temperatures[i].append(temp)
                statuses[i].append(int(status))
        return temperatures, statuses

    # def simulate_one_shot_shedding(self, shed_kw, time_window, duration_minutes):
    #     import copy
    #
    #     # Ensure baseline state is up-to-date
    #     if self.fleet is None:
    #         self.setup()
    #
    #     off_indices = set()
    #     temperatures, statuses = init_storage()
    #
    #     for t in range(duration_minutes):
    #         # --- 1) Baseline copy to see what WOULD be on ---
    #         fleet_copy = copy.deepcopy(self.fleet)
    #         for h in fleet_copy.heaters:
    #             h.control_heating_element()
    #         baseline_statuses = [h.heating_element_status for h in fleet_copy.heaters]
    #         baseline_powers = [
    #             (i, h.heating_element_power_w / 1000.0, h.temperature)
    #             for i, (h, s) in enumerate(zip(fleet_copy.heaters, baseline_statuses))
    #             if s
    #         ]
    #
    #         # --- 2) At window start, pick which to turn OFF ---
    #         if t == time_window[0]:
    #             # Sort ON heaters by descending temperature
    #             baseline_powers.sort(key=lambda x: -x[2])
    #             cum = 0.0
    #             for i, p, temp in baseline_powers:
    #                 if cum < shed_kw:
    #                     off_indices.add(i)
    #                     cum += p
    #                 else:
    #                     break
    #             print(f"Shedding {cum:.1f} kW by switching OFF {len(off_indices)} heaters")
    #
    #         # --- 3) Apply control for this minute ---
    #         if time_window[0] <= t < time_window[1]:
    #             # In the window, force OFF our selected heaters
    #             for i, heater in enumerate(self.fleet.heaters):
    #                 if i in off_indices:
    #                     heater.set_heating_element_status(False)
    #                 else:
    #                     # mirror baseline for the others
    #                     heater.set_heating_element_status(baseline_statuses[i])
    #         else:
    #             # Outside window: exactly baseline
    #             for i, heater in enumerate(self.fleet.heaters):
    #                 heater.set_heating_element_status(baseline_statuses[i])
    #
    #         # --- 4) Advance all heaters one step ---
    #         for i, heater in enumerate(self.fleet.heaters):
    #             temp, status = heater.step(t, skip_control=True)
    #             temperatures[i].append(temp)
    #             statuses[i].append(int(status))
    #
    #     return np.array(temperatures), np.array(statuses)


if __name__ == "__main__":

    from utils.data_handling import ewh_characteristics, ewh_water_consumption, ewh_power_consumption
    from utils.plotting import PlotManager
    import pandas as pd
    import numpy as np
    import copy
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

    # baseline_sim = copy.deepcopy(sim)
    # temperatures, statuses = baseline_sim.simulate_baseline(duration_minutes=duration_minutes)
    # power_profiles = np.array([
    #     statuses[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
    #     for i in range(n_heaters)
    # ])
    # agg_power = power_profiles.sum(axis=0)
    # window_start, window_end = time_window
    # agg_power_window = agg_power[window_start:window_end]
    # # print(agg_power_window)
    # plotter.plot_power_profiles(time_index, power_profiles, agg_power, fig1 = False, fig2 = True, ewh_power_consumption=ewh_power_consumption)
    # # plotter.plot_temperature(time_index, temperatures)
    # # plotter.plot_status(time_index, statuses)
    # # plotter.plot_boxplot(data['characteristics'])



    # capped_sim = copy.deepcopy(sim)
    # temperatures_capped, statuses_capped = capped_sim.simulate_power_capped(duration_minutes=duration_minutes,
    #                                                                         cap_kw=cap_kw, time_window=time_window)
    # power_profiles_capped = np.array([
    #     statuses_capped[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
    #     for i in range(n_heaters)
    # ])
    # #
    # #
    # agg_power_capped = power_profiles_capped.sum(axis=0)
    # plotter.plot_aggregate_power_comparison(
    #     time_index, agg_power, agg_power_capped,
    #     label_baseline="Baseline (no capping)",
    #     label_shedding=f"With power cap ({cap_kw} kW)"
    # )
    #
    #
    # shed_sim = copy.deepcopy(sim)
    # temperatures_shed, statuses_shed = shed_sim.simulate_operator_shedding(duration_minutes=duration_minutes,
    #                                                                        shed_kw=shed_kw, time_window=time_window)
    # power_profiles_shed = np.array([
    #     statuses_shed[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
    #     for i in range(n_heaters)
    # ])
    # agg_power_shed = power_profiles_shed.sum(axis=0)
    # plotter.plot_aggregate_power_comparison(
    #     time_index, agg_power, agg_power_shed,
    #     label_baseline="Baseline (no shedding)",
    #     label_shedding=f"Operator shedding ({shed_kw} kW)"
    # )

    # alt_shed_sim = copy.deepcopy(sim)
    # temperatures_alt_shed, statutes_alt_shed = alt_shed_sim.simulate_operator_shedding_alternative(
    #     duration_minutes=duration_minutes, shed_kw=shed_kw, time_window=time_window)
    # # Calculate power profiles
    # power_profiles_alt_shed = np.array([
    #     statutes_alt_shed[i] * data['characteristics'].iloc[i]['heating_element_power_w'] / 1000.0
    #     for i in range(n_heaters)
    # ])
    # agg_power_alt_shed = power_profiles_alt_shed.sum(axis=0)
    #
    #
    # # Plot results
    #
    # # plotter.plot_power_profiles(time_index, power_profiles, agg_power, ewh_power_consumption=ewh_power_consumption)
    # # plotter.plot_temperature(time_index, temperatures)
    # # plotter.plot_status(time_index, statuses)
    # # plotter.plot_boxplot(data['characteristics'])
    #
    # plotter.plot_aggregate_power_comparison(
    #     time_index, agg_power, agg_power_alt_shed,
    #     label_baseline="Baseline (no shedding)",
    #     label_shedding=f"Alternative operator shedding ({shed_kw} kW)"
    # )

    # Run enhanced power capping simulation
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
    # print(f"Power Capping Results (cap: {cap_kw} kW, window: {window_start}-{window_end} min):")
    # print(f"Energy saved during window: {metrics['energy_saved_kwh']:.2f} kWh")
    # print(f"Rebound factor: {metrics['rebound_factor']:.2f} x baseline peak")

    # Create custom visualization comparing capped vs baseline power
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
