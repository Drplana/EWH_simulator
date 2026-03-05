from os import TMP_MAX

import pandas as pd
from water_consumption.stochastic_profile import StochasticDHWProfile
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
import plotly.graph_objects as go


class ElectricWaterHeater:
    def __init__(
        self,
        n_users,
        demand_profile=None,
        nday=1,
        volume_liters=None,
        diameter_m=None,
        height_m=None,
        density_kgm3=997,  # Water at ~25°C If the user wants more accurate results can use coolprops.
        heating_element_power_w=2000,
        initial_temp_C=50,
        room_temp_C=20,
        cold_water_inlet_temp_C=15,
        max_temp_C=65,
        min_temp_C=40,
        heat_transfer_coeff=0.8,  # W/(m²/K)
        heating_element_status=False,
        hysteresis=2,  # Hysteresis for control
        control_strategy = "thermostat"
    ):
        if demand_profile is not None:
            self.demand_profile = demand_profile
        else:
            self.demand_profile = StochasticDHWProfile(nday=nday, n_users=n_users).DHW_load_gen()
        self.n_users = n_users

        if diameter_m is not None and height_m is not None:
            self.surface_area_m2 = self.calculate_surface_area(diameter_m, height_m)
        else:
            self.surface_area_m2 = None

    # Volume calculation
        if volume_liters is not None:
            self.volume_liters = volume_liters
        elif diameter_m is not None and height_m is not None:
            self.volume_liters = self.calculate_volume_from_dimensions(diameter_m, height_m)
        else:
            self.volume_liters = max(50, n_users * 50)
        self.diameter_m = diameter_m
        self.height_m = height_m
        # mass calculation should be similar to volume number
        self.mass_kg = self.volume_liters / 1000 * density_kgm3
        # Heating element
        self.heating_element_power_w = heating_element_power_w
        self.heating_element_status = heating_element_status

        # Temperatures and conditions
        self.initial_temp_C = initial_temp_C
        self.current_temp_C = initial_temp_C
        self.room_temp_C = room_temp_C
        self.cold_water_inlet_temp_C = cold_water_inlet_temp_C
        self.max_temp_C = max_temp_C # In practice it is set by the user, but maximum 80 C.
        self.min_temp_C = min_temp_C # Minimum temperature to avoid user discomfort could be 38 C
        self.heat_transfer_coeff = heat_transfer_coeff
        self.timestep_s = 60
        self.nday = nday
        self.water_thermal_capacity = 4186  # J/(kg·K) for water
        # Control strategy
        self.control_strategy = control_strategy
        self.hysteresis = hysteresis

    @staticmethod
    def calculate_volume_from_dimensions(diameter_m, height_m):
        """Calculate volume in liters from diameter and height (cylinder)."""
        volume_m3 = np.pi * (diameter_m / 2) ** 2 * height_m
        return volume_m3 * 1000  # Convert to liters
    @staticmethod
    def calculate_surface_area(diameter_m, height_m):
        """Calculate surface area in m² from diameter and height (cylinder)."""
        radius = diameter_m / 2
        lateral_area = 2 * np.pi * radius * height_m
        top_bottom_area = 2 * np.pi * radius ** 2
        return lateral_area + top_bottom_area

    def set_heating_element_status(self, status: bool):
        self.heating_element_status = status

    def calculate_energy_consumption(self, timestep_s):
        """
        Calculate the energy consumption of the electric water heater over a given time step.

        Parameters:
        - timestep_s: Time step in seconds.

        Returns:
        - Total energy consumption in kWh.
        """
        # Energy consumed by the heating element (if active)
        if self.heating_element_status:
            energy_heating = self.heating_element_power_w * timestep_s  # in joules
        else:
            energy_heating = 0

        # Heat loss due to conduction
        if self.surface_area_m2 is not None:
            delta_temp = self.current_temp_C - self.room_temp_C  # Temperature difference in K
            heat_loss = self.heat_transfer_coeff * self.surface_area_m2 * delta_temp * timestep_s  # in joules
        else:
            heat_loss = 0

        # Total energy consumption (convert joules to kWh)
        total_energy_joules = energy_heating + heat_loss
        total_energy_kwh = total_energy_joules / (3600 * 1000)  # Convert to kWh

        return total_energy_kwh

    # def control_heating_element_with_hysteresis(self):
    #     if self.current_temp_C >= self.max_temp_C + self.hysteresis:
    #         self.set_heating_element_status(False)
    #     elif self.current_temp_C <= self.min_temp_C - self.hysteresis:
    #         self.set_heating_element_status(True)

    def control_heating_element_with_hysteresis(self, hysteresis=2):
        if self.current_temp_C >= self.max_temp_C + hysteresis:
            self.set_heating_element_status(False)
        elif self.current_temp_C <= self.min_temp_C - hysteresis:
            self.set_heating_element_status(True)
    def control_heating_element_with_thermostat(self):
        """
        Control the heating element based on thermostat settings.
        If the current temperature is above max_temp_C, turn off the heating element.
        If the current temperature is below min_temp_C, turn on the heating element.
        """
        if self.current_temp_C >= self.max_temp_C:
            self.set_heating_element_status(False)
        elif self.current_temp_C <= self.min_temp_C:
            self.set_heating_element_status(True)


    def control_heating_element(self):
        if self.control_strategy == "hysteresis":
            self.control_heating_element_with_hysteresis()
        elif self.control_strategy == "thermostat":
            self.control_heating_element_with_thermostat()



    def simulate_heating_no_water_draws(self, duration_minutes):
        """
        Simulate the heating process over a given duration.

        Parameters:
        - duration_minutes: Total simulation time in minutes.

        Returns:
        - A list of temperatures at each timestep.
        """
        num_steps = int(duration_minutes * 60 / self.timestep_s)
        temperatures = [self.current_temp_C]
        statuses = [self.heating_element_status]
        demand = self.demand_profile['total'].values

        for _ in range(num_steps):
            if self.current_temp_C >= self.max_temp_C:
                self.set_heating_element_status(False)
            elif self.current_temp_C <= self.min_temp_C:
                self.set_heating_element_status(True)

            if self.heating_element_status:
                energy_heating = self.heating_element_power_w * self.timestep_s  # in joules
                temp_increase = energy_heating / (self.mass_kg * self.water_thermal_capacity)  # 4186 J/(kg·K) for water
            else:
                temp_increase = 0

            if self.surface_area_m2 is not None:
                delta_temp = self.current_temp_C - self.room_temp_C
                heat_loss = self.heat_transfer_coeff * self.surface_area_m2 * delta_temp * self.timestep_s  # in joules
                temp_decrease = heat_loss / (self.mass_kg * self.water_thermal_capacity )
            else:
                temp_decrease = 0

            self.current_temp_C += temp_increase - temp_decrease
            temperatures.append(self.current_temp_C)
            statuses.append(self.heating_element_status)
        return temperatures, statuses

    def simulate_heating_old(self, duration_minutes):
        num_steps = int(duration_minutes * 60 / self.timestep_s)
        temperatures = [self.current_temp_C]
        statuses = [self.heating_element_status]
        demand = self.demand_profile['total'].values  # L/s, length should match simulation steps

        for step in range(num_steps):
            # 1. Water draw for this timestep (liters)
            if step < len(demand):
                water_draw_liters = demand[step] * self.timestep_s
            else:
                water_draw_liters = 0

            # 2. Calculate temperature drop due to water draw
            if water_draw_liters > 0:
                # Mix cold water at room_temp_C into the tank
                mixed_mass = self.mass_kg
                draw_mass = water_draw_liters / 1000 * 997  # kg
                new_temp = (
                                   (self.current_temp_C * (mixed_mass - draw_mass)) +
                                   (self.cold_water_inlet_temp_C * draw_mass)
                           ) / mixed_mass
                self.current_temp_C = new_temp

            # 3. Control strategy for resistor
            self.control_heating_element()

                # 4. Heating element
            if self.heating_element_status:
                energy_heating = self.heating_element_power_w * self.timestep_s
                temp_increase = energy_heating / (self.mass_kg * self.water_thermal_capacity)
            else:
                temp_increase = 0

            # 5. Ambient heat loss
            if self.surface_area_m2 is not None:
                delta_temp = self.current_temp_C - self.room_temp_C
                heat_loss = self.heat_transfer_coeff * self.surface_area_m2 * delta_temp * self.timestep_s
                temp_decrease = heat_loss / (self.mass_kg * self.water_thermal_capacity)
            else:
                temp_decrease = 0

            self.current_temp_C += temp_increase - temp_decrease
            temperatures.append(self.current_temp_C)
            statuses.append(self.heating_element_status)
        return temperatures, statuses

    def step(self, timestep_idx=None, skip_control= False):
        demand = self.demand_profile['total'].values
        if timestep_idx is not None and timestep_idx < len(demand):
            water_draw_liters = demand[timestep_idx] * self.timestep_s
        else:
            water_draw_liters = 0

        if water_draw_liters > 0:
            mixed_mass = self.mass_kg
            draw_mass = water_draw_liters / 1000 * 997
            new_temp = (
                (self.current_temp_C * (mixed_mass - draw_mass)) +
                (self.cold_water_inlet_temp_C * draw_mass)
            ) / mixed_mass
            self.current_temp_C = new_temp

        if not skip_control:
            self.control_heating_element()

        if self.heating_element_status:
            energy_heating = self.heating_element_power_w * self.timestep_s
            temp_increase = energy_heating / (self.mass_kg * self.water_thermal_capacity)
        else:
            temp_increase = 0

        if self.surface_area_m2 is not None:
            delta_temp = self.current_temp_C - self.room_temp_C
            heat_loss = self.heat_transfer_coeff * self.surface_area_m2 * delta_temp * self.timestep_s
            temp_decrease = heat_loss / (self.mass_kg * self.water_thermal_capacity)
        else:
            temp_decrease = 0

        self.current_temp_C += temp_increase - temp_decrease
        return self.current_temp_C, self.heating_element_status

    def simulate_heating(self, duration_minutes):
        num_steps = int(duration_minutes * 60 / self.timestep_s)
        temperatures = [self.current_temp_C]
        statuses = [self.heating_element_status]
        for step in range(num_steps):
            temp, status = self.step(step)
            temperatures.append(temp)
            statuses.append(status)
        return temperatures, statuses


if __name__ =="__main__":
    ewh = ElectricWaterHeater(max_temp_C = 80, min_temp_C=75, initial_temp_C=80, n_users=2, nday=1, volume_liters=None, diameter_m=0.210, height_m=0.930)
    duration_minutes = ewh.nday * 24 * 60
    temperatures, statuses = ewh.simulate_heating(duration_minutes)
    timesteps = np.arange(len(temperatures)) * ewh.timestep_s / 60  # minutes

    # Create a DatetimeIndex starting at midnight
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_index = pd.date_range(start=start_time, periods=len(temperatures), freq=f'{ewh.timestep_s // 60}min')
    water_draw = ewh.demand_profile['total'].values
    water_draw = np.pad(water_draw, (0, len(temperatures) - len(water_draw)), 'constant')
    df_water = pd.DataFrame({"Water Consumption (L/s)": water_draw}, index=time_index)
    df_temp = pd.DataFrame({"Temperature (°C)": temperatures}, index=time_index)
    df_status = pd.DataFrame({"Resistor Status (On=1, Off=0)": [int(s) for s in statuses]}, index=time_index)
    appliance_names = ['sinkA', 'sinkB', 'shower', 'bath', 'total']
    appliance_data = {}
    for name in appliance_names:
        arr = ewh.demand_profile[name].values
        arr = np.pad(arr, (0, len(temperatures) - len(arr)), 'constant')
        appliance_data[name] = arr

    df_appliances = pd.DataFrame({
        'Sink A (L/s)': appliance_data['sinkA'],
        'Sink B (L/s)': appliance_data['sinkB'],
        'Shower (L/s)': appliance_data['shower'],
        'Bath (L/s)': appliance_data['bath'],
        'Total (L/s)': appliance_data['total'],
    }, index=time_index)


    st.title("Electric Water Heater Simulation")
    st.line_chart(df_temp)
    st.line_chart(df_status)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=appliance_data['sinkA'], stackgroup='one', name='Sink A'))
    fig.add_trace(go.Scatter(x=time_index, y=appliance_data['sinkB'], stackgroup='one', name='Sink B'))
    fig.add_trace(go.Scatter(x=time_index, y=appliance_data['shower'], stackgroup='one', name='Shower'))
    fig.add_trace(go.Scatter(x=time_index, y=appliance_data['bath'], stackgroup='one', name='Bath'))
    fig.add_trace(go.Scatter(x=time_index, y=appliance_data['total'], mode='lines', name='Total', line=dict(color='darkblue', width=2)))

    fig.update_layout(title='DHW Profile by Appliance (Stacked)', xaxis_title='Time', yaxis_title='Flow [L/s]')
    st.plotly_chart(fig)