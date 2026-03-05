from water_consumption.stochastic_profile import StochasticDHWProfile, StochasticDHWProfile_1
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Create a profile for 1 day, 3 users
new_profile = StochasticDHWProfile(nday=1, n_users=3)
df_new = new_profile.DHW_load_gen()

# If you have access to the old profile generator
old_profile = StochasticDHWProfile_1(nday=1, n_users=3)
df_old = old_profile.DHW_load_gen()


