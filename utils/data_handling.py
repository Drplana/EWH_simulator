# Load the CSV
import os
import pandas as pd
import streamlit as st

ewh_characteristics_path = '/home/david/UnleashEWH/utils/TEST_digital_twin_WH_charact_1.csv'

ewh_characteristics = pd.read_csv(ewh_characteristics_path, sep=';')# Adjust separator if needed


ewh_characteristics = ewh_characteristics.rename(columns={
'Volume (L)': 'volume_liters',
'Electric Power (W)': 'heating_element_power_w',
'Height (m)': 'height_m',
'Diameter (m)': 'diameter_m'
})
# print(ewh_characteristics)
# print(ewh_characteristics.head())

ewh_water_consumption_path = '/home/david/UnleashEWH/utils/TEST_digital_twin_WH_timeseries_1.csv'
cols = pd.read_csv(ewh_water_consumption_path, sep=';', nrows=0).columns
usecols = [cols[0]] + [col for col in cols if col.startswith('vdot')]
ewh_water_consumption = pd.read_csv(ewh_water_consumption_path, sep=';', usecols=usecols)
ewh_power_consumption = pd.read_csv(ewh_water_consumption_path, sep=';', usecols=['cumulated_power_kW'])

# print(ewh_water_consumption.head())


