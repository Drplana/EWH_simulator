from ewh_model.single_ewh import ElectricWaterHeater
import plotly.graph_objects as go
import numpy as np
import random
import pandas as pd
import plotly.express as px
class ElectricWaterHeaterFleet:
    def __init__(self, n_heaters, n_users_per_heater=2, **ewh_kwargs):
        self.n_heaters = n_heaters
        self.ewh_kwargs = ewh_kwargs
        self.n_users_per_heater = n_users_per_heater
        self.heaters = [
            ElectricWaterHeater(n_users=n_users_per_heater, **ewh_kwargs)
            for _ in range(n_heaters)
        ]

    def get_active_heaters(self):
        """
        Returns a list of heaters that are currently ON.
        """
        return [h for h in self.heaters if h.heating_element_status]

    def get_inactive_heaters(self):
        """
        Returns a list of heaters that are currently OFF.
        """
        return [h for h in self.heaters if not h.heating_element_status]

    def sort_active_heaters_by_temperature(self):
        """
        Returns a list of active heaters (ON status) sorted by their
        current temperature in descending order.
        """
        heaters_on = self.get_active_heaters()
        return sorted(heaters_on, key=lambda h: h.current_temp_C, reverse=True)



    def apply_power_shedding(self, power_cap_kw):
        """
        Reduce fleet power consumption to be below power_cap_kw by turning off
        heaters starting with those at highest temperatures.

        Parameters:
            power_cap_kw (float): Maximum allowed power consumption in kW

        Returns:
            dict: Statistics about the shedding operation
        """
        # 1. Find all heaters that are ON
        heaters_on = self.get_active_heaters()
        total_on_power = sum(h.heating_element_power_w for h in heaters_on) / 1000.0  # kW

        # 2. If already under cap, do nothing
        if total_on_power <= power_cap_kw:
            return {"shed_count": 0, "initial_power_kw": total_on_power, "final_power_kw": total_on_power}

        # 3. Sort ON heaters by temperature (highest first)
        sorted_heaters = self.sort_active_heaters_by_temperature()

        # 4. Turn OFF heaters with highest temp until under cap
        shed_count = 0
        for h in sorted_heaters:
            if total_on_power <= power_cap_kw:
                break
            h.set_heating_element_status(False)
            total_on_power -= h.heating_element_power_w / 1000.0
            shed_count += 1

        return {
            "shed_count": shed_count,
            "initial_power_kw": sum(h.heating_element_power_w for h in heaters_on) / 1000.0,
            "final_power_kw": total_on_power
        }



    def apply_power_shutdown(self, shutdown_power_kw):
        # 1. Find all heaters that are ON
        heaters_on = [h for h in self.heaters if h.heating_element_status]
        # 2. Sort ON heaters by temperature (highest first)
        heaters_on.sort(key=lambda h: h.current_temp_C, reverse=True)
        # 3. Turn OFF heaters until shutdown_power_kw is reached
        shed_power = 0.0
        for h in heaters_on:
            if shed_power >= shutdown_power_kw:
                break
            h.set_heating_element_status(False)
            shed_power += h.heating_element_power_w / 1000.0  # kW



    def apply_power_shutdown_given_statuses(self, shutdown_power_kw, future_statuses):
        """
        Turn OFF enough heaters (from those predicted to be ON in future_statuses)
        to shed shutdown_power_kw. Only heaters with future_statuses[i]==True are eligible.
        """
        # Find indices of heaters predicted to be ON
        on_indices = [i for i, status in enumerate(future_statuses) if status]
        # Sort these heaters by current_temp_C (highest first)
        sorted_on = sorted(on_indices, key=lambda i: self.heaters[i].current_temp_C, reverse=True)
        shed_power = 0.0
        for i in sorted_on:
            if shed_power >= shutdown_power_kw:
                break
            self.heaters[i].set_heating_element_status(False)
            shed_power += self.heaters[i].heating_element_power_w / 1000.0  # kW


    def create_randomized_heaters(self, n_heaters=None):
        if n_heaters is not None:
            self.n_heaters = n_heaters
        volume_probs = {
            1: [0.7, 0.2, 0.05, 0.03, 0.01, 0.01],  # 1 user: mostly 120L
            2: [0.2, 0.5, 0.2, 0.07, 0.02, 0.01],  # 2 users: mostly 150L
            3: [0.05, 0.2, 0.5, 0.2, 0.04, 0.01],  # 3 users: mostly 180L
            4: [0.01, 0.05, 0.2, 0.5, 0.2, 0.04],  # 4 users: mostly 210L
            5: [0.01, 0.01, 0.04, 0.2, 0.5, 0.24],  # 5 users: mostly 250/280L
        }
        Volumes = [120, 150, 180, 210, 250, 280]  # liters
        Power_range = [2000, 2200, 2400, 2600, 2800, 3000]  # W
        height_mean = 1.0  # m
        height_std = 0.2  # m

        household_sizes = [1, 2, 3, 4, 5]
        probabilities = [0.339, 0.318, 0.155, 0.131, 0.057]
        self.heaters = []

        for _ in range(self.n_heaters):
            n_users = np.random.choice(household_sizes, p=probabilities)
            probs = volume_probs[n_users]
            volume_liters = np.random.choice(Volumes, p=probs)
            # min_volume_liters = max(50 * n_users, 120)
            # possible_volumes = [v for v in Volumes if v >= min_volume_liters]
            # volume_liters = random.choice(possible_volumes)
            power = random.choice(Power_range)
            height = max(0.5, np.random.normal(height_mean, height_std))
            diameter = np.sqrt(4 * (volume_liters / 1000) / (np.pi * height))

            heater = ElectricWaterHeater(
                n_users=n_users,
                volume_liters=volume_liters,
                diameter_m=diameter,
                height_m=height,
                heating_element_power_w=power,
                **self.ewh_kwargs  # Pass all other parameters (temperatures, etc.)
            )
            self.heaters.append(heater)


# Example usage
if __name__ == "__main__":
    n_heaters = 1000
    # n_users_per_heater = 2
    # initial_temp_C = 80
    # max_temp_C = 80
    # min_temp_C = 75
    #
    # fleet = ElectricWaterHeaterFleet(
    #     n_heaters=n_heaters,
    #     n_users_per_heater=n_users_per_heater,
    #     initial_temp_C=initial_temp_C,
    #     max_temp_C=max_temp_C,
    #     min_temp_C=min_temp_C,
    #     nday=1
    # )
    # n_heaters = 1000
    # # fleet = ElectricWaterHeaterFleet(
    # #     n_heaters=n_heaters,
    # #     n_users_per_heater=2,
    # #     initial_temp_C=80,
    # #     max_temp_C=80,
    # #     min_temp_C=75,
    # #     nday=1
    # # )
    #
    # # Randomize heater parameters
    # fleet = ElectricWaterHeaterFleet(n_heaters=n_heaters)
    # fleet.create_randomized_heaters(n_heaters=n_heaters)

    # Run the simulation
    # results = fleet.simulate_all(duration_minutes=24 * 60)
    # fleet.heaters = []
    # fleet.create_randomized_heaters()

    # results = fleet.simulate_all(duration_minutes=24*60)

    # n_heaters = len(results)
    # timesteps = len(results[0]['statuses'])
    # timestep_s = fleet.heaters[0].timestep_s
    # time_index = np.arange(timesteps) * timestep_s / 60  # minutes
    #
    # # Collect power profiles
    # power_profiles = []
    # for i, ewh in enumerate(fleet.heaters):
    #     status = np.array(results[i]['statuses'][:-1])  # match length
    #     power = status * ewh.heating_element_power_w / 1000.0  # kW
    #     power_profiles.append(power)
    #
    # power_profiles = np.array(power_profiles)  # Now it's a NumPy array
    # agg_power = power_profiles.sum(axis=0)  # Aggregated power

    # print(f"Number of heaters: {n_heaters}")
    # print(f"Power profiles shape: {power_profiles.shape}")
    # print(f"First power profile: {power_profiles[0][:10]}")
    # print(f"Aggregated power (first 10): {agg_power[:10]}")


    # for i, ewh in enumerate(fleet.heaters):
    #     print(f"Heater {i} demand (first 10): {ewh.demand_profile['total'].values[:10]}")
    #     print(f"Initial temp: {ewh.initial_temp_C}, Min temp: {ewh.min_temp_C}, Max temp: {ewh.max_temp_C}")

    # for i, ewh in enumerate(fleet.heaters):
    #     status = np.array(results[i]['statuses'][:-1])  # match length
    #     power = status * ewh.heating_element_power_w /1000 # W
    #     power_profiles.append(power)

    # power_profiles = np.array(power_profiles)  # shape: (n_heaters, timesteps)
    # agg_power = power_profiles.sum(axis=0)  # aggregated power


    # power_profiles_kw = power_profiles / 1000.0
    # agg_power_kw = agg_power / 1000.0

    # Plot individual power profiles

    # fig1 = go.Figure()
    # for i in range(n_heaters):
    #     fig1.add_trace(go.Scatter(x=time_index, y=power_profiles[i], mode='lines', name=f'EWH {i + 1}'))
    # fig1.update_layout(title='Power Consumption Profiles for Each EWH', xaxis_title='Time [min]',
    #                    yaxis_title='Power [kW]')
    #
    # # Plot aggregated power curve
    # fig2 = go.Figure()
    # fig2.add_trace(
    #     go.Scatter(x=time_index, y=agg_power, mode='lines', name='Aggregated Power', line=dict(color='red', width=3)))
    # fig2.update_layout(title='Aggregated Power Consumption Curve', xaxis_title='Time [min]', yaxis_title='Power [W]')
    #
    # # Show with Streamlit (or plt.show() if not using Streamlit)
    # import streamlit as st
    #
    # # Collect data from the fleet
    # user_volume_data = [
    #     {"n_users": h.n_users, "volume_liters": h.volume_liters}
    #     for h in fleet.heaters
    # ]
    # df = pd.DataFrame(user_volume_data)
    #
    # # Show statistics in Streamlit
    # st.write("### Tank Volume Statistics by Number of Users")
    # stats = df.groupby("n_users")["volume_liters"].describe()
    # st.dataframe(stats)
    #
    # # Plot boxplot: tank volume by number of users
    # fig = px.box(
    #     df, x="n_users", y="volume_liters", points="all",
    #     title="Tank Volume Distribution by Number of Users",
    #     labels={"n_users": "Number of Users", "volume_liters": "Tank Volume (L)"}
    # )
    # count_table = df.groupby(['n_users', 'volume_liters']).size().unstack(fill_value=0)
    #
    # st.write("### Number of Heaters by Number of Users and Tank Volume (L)")
    # st.dataframe(count_table)
    # st.plotly_chart(fig)
    #
    # st.plotly_chart(fig1)
    # st.plotly_chart(fig2)
