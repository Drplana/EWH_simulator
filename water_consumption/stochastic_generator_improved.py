# import numpy as np
# import random
# import xarray as xr
# import matplotlib.pyplot as plt
#
# # ------------------------------------------------------------------------------------
# # 1) Household sampling utilities
# # ------------------------------------------------------------------------------------
# BELGIAN_HOUSEHOLD_TYPES = [
#     ("one_person",               0.361, 1, {}),
#     ("married_no_children",      0.182, 2, {}),
#     ("married_with_children",    0.187, 4, {"has_children": True}),
#     ("cohabiting_no_children",   0.066, 2, {}),
#     ("cohabiting_with_children", 0.082, 3, {"has_children": True}),
#     ("lone_parent",              0.099, 2, {"has_children": True}),
#     ("other",                    0.022, 3, {}),
# ]
#
# def sample_household_type():
#     """
#     Randomly pick a Belgian household type (name, average users, demographics dict),
#     based on the probabilities in BELGIAN_HOUSEHOLD_TYPES.
#     """
#     types, probs, sizes, demos = zip(*BELGIAN_HOUSEHOLD_TYPES)
#     probs = np.array(probs)
#     probs = probs / probs.sum()
#     idx = np.random.choice(len(types), p=probs)
#     return types[idx], sizes[idx], demos[idx]
#
#
# # ------------------------------------------------------------------------------------
# # 2) Water-use profile configuration: per-region, per-season, 24-hour probability vectors
# # ------------------------------------------------------------------------------------
# WATER_USE_PROFILES = {
#     "BE": {
#         "winter": {
#             "shower": [0.013, 0.006, 0.004, 0.004, 0.009, 0.020, 0.050, 0.062,
#                        0.070, 0.065, 0.050, 0.040, 0.035, 0.030, 0.028, 0.028,
#                        0.032, 0.045, 0.050, 0.050, 0.045, 0.033, 0.025, 0.018],
#             "bath":   [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.010, 0.015,
#                        0.020, 0.018, 0.015, 0.010, 0.008, 0.005, 0.005, 0.005,
#                        0.012, 0.020, 0.025, 0.030, 0.030, 0.025, 0.020, 0.010],
#             "sink":   [0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.060,
#                        0.065, 0.060, 0.055, 0.050, 0.049, 0.047, 0.044, 0.045,
#                        0.050, 0.070, 0.075, 0.070, 0.060, 0.052, 0.042, 0.030],
#             "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.010, 0.020, 0.030, 0.020, 0.010, 0.000, 0.000],
#         },
#         "spring": {
#             "shower": [0.011, 0.005, 0.003, 0.003, 0.008, 0.018, 0.045, 0.055,
#                        0.060, 0.055, 0.045, 0.035, 0.030, 0.025, 0.023, 0.023,
#                        0.028, 0.040, 0.045, 0.045, 0.040, 0.030, 0.022, 0.015],
#             "bath":   [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.008, 0.012,
#                        0.015, 0.013, 0.012, 0.008, 0.007, 0.005, 0.005, 0.005,
#                        0.010, 0.018, 0.020, 0.025, 0.025, 0.020, 0.015, 0.008],
#             "sink":   [0.014, 0.007, 0.005, 0.005, 0.007, 0.018, 0.044, 0.058,
#                        0.060, 0.058, 0.052, 0.048, 0.047, 0.045, 0.042, 0.043,
#                        0.046, 0.064, 0.070, 0.065, 0.055, 0.048, 0.040, 0.028],
#             "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.008, 0.015, 0.020, 0.015, 0.008, 0.000, 0.000],
#         },
#         "summer": {
#             "shower": [0.010, 0.004, 0.003, 0.003, 0.008, 0.017, 0.042, 0.052,
#                        0.058, 0.052, 0.042, 0.032, 0.028, 0.023, 0.020, 0.022,
#                        0.025, 0.035, 0.040, 0.040, 0.040, 0.035, 0.025, 0.017],
#             "bath":   [0.001, 0.000, 0.000, 0.000, 0.001, 0.004, 0.007, 0.010,
#                        0.012, 0.010, 0.009, 0.006, 0.005, 0.004, 0.004, 0.004,
#                        0.008, 0.015, 0.018, 0.020, 0.020, 0.018, 0.015, 0.008],
#             "sink":   [0.013, 0.007, 0.005, 0.005, 0.006, 0.018, 0.042, 0.056,
#                        0.058, 0.056, 0.048, 0.046, 0.045, 0.042, 0.040, 0.042,
#                        0.044, 0.060, 0.068, 0.064, 0.052, 0.046, 0.038, 0.026],
#             "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.006, 0.010, 0.015, 0.010, 0.006, 0.000, 0.000],
#         },
#         "autumn": {
#             "shower": [0.012, 0.006, 0.004, 0.004, 0.009, 0.019, 0.048, 0.060,
#                        0.065, 0.060, 0.050, 0.040, 0.036, 0.030, 0.028, 0.028,
#                        0.030, 0.043, 0.048, 0.048, 0.044, 0.032, 0.024, 0.017],
#             "bath":   [0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.009, 0.014,
#                        0.018, 0.016, 0.014, 0.010, 0.009, 0.006, 0.006, 0.006,
#                        0.014, 0.022, 0.028, 0.032, 0.032, 0.028, 0.022, 0.010],
#             "sink":   [0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.058,
#                        0.062, 0.058, 0.053, 0.049, 0.048, 0.046, 0.044, 0.045,
#                        0.048, 0.068, 0.072, 0.068, 0.058, 0.050, 0.042, 0.030],
#             "dishwasher": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
#                            0.000, 0.009, 0.015, 0.025, 0.015, 0.009, 0.000, 0.000],
#         },
#     },
#     "NL": {
#         "winter":    {"shower": [0.012]*24, "bath": [0.008]*24, "sink": [0.014]*24, "dishwasher": [0.000]*24},
#         "spring":    {"shower": [0.010]*24, "bath": [0.006]*24, "sink": [0.012]*24, "dishwasher": [0.000]*24},
#         "summer":    {"shower": [0.009]*24, "bath": [0.005]*24, "sink": [0.011]*24, "dishwasher": [0.000]*24},
#         "autumn":    {"shower": [0.011]*24, "bath": [0.007]*24, "sink": [0.013]*24, "dishwasher": [0.000]*24},
#     },
# }
#
# # Seasonal scaling factors (relative to annual average)
# SEASONAL_SCALING = {
#     "winter": 1.12,   # +12 % in winter
#     "spring": 1.00,   # baseline
#     "summer": 0.87,   # –13 % in summer
#     "autumn": 1.00,   # baseline
# }
#
#
# # ------------------------------------------------------------------------------------
# # 3) Final DHW class with all enhancements
# # ------------------------------------------------------------------------------------
# class StochasticDHWProfileFinal:
#     """
#     Stochastic DHW profile generator with:
#       • Region‐ and season‐aware start‐probabilities for shower, bath, sink, dishwasher
#       • Seasonal scaling (±12–13 %) of probabilities
#       • Correct unit conversions → instantaneous L/min at each minute
#       • Calibration to a target daily volume per person (default 50 L/p·d)
#       • Optional energy output in kWh (ΔT = 50 K)
#     """
#
#     def __init__(
#         self,
#         nday=1,
#         household_type=None,
#         n_users=None,
#         demographics=None,
#         seed=None,
#         fixture_params=None,
#         region="BE",
#         season="winter",
#         target_L_per_person=50,
#     ):
#         # 1) Possibly sample a household type (if none given)
#         if household_type is None:
#             household_type, n_users, demographics = sample_household_type()
#
#         self.household_type = household_type
#         self.nday = nday
#         self.n_users = n_users if n_users is not None else 1
#         self.demographics = demographics or {}
#         self.fixture_params = fixture_params or {}
#         self.region = region
#         self.season = season
#         self.target_L_per_person = target_L_per_person
#
#         # 2) Random seeds for reproducibility
#         if seed is not None:
#             np.random.seed(seed)
#             random.seed(seed)
#
#         # 3) Load per-hour probabilities and apply seasonal scaling
#         self._load_time_of_day_probs()
#
#         # 4) Build fixture definitions and apply overrides
#         self._set_fixtures()
#
#     def _load_time_of_day_probs(self):
#         """Lookup and scale the 24‐hour probability lists for this region & season."""
#         region_block = WATER_USE_PROFILES.get(self.region)
#         if region_block is None:
#             raise KeyError(f"No water‐use profile for region '{self.region}'.")
#
#         season_block = region_block.get(self.season)
#         if season_block is None:
#             raise KeyError(f"No water‐use profile for season '{self.season}' in region '{self.region}'.")
#
#         # Extract hourly probability vectors (as numpy arrays)
#         self.hourly_shower_probs = np.array(season_block["shower"], dtype=float)
#         self.hourly_bath_probs = np.array(season_block["bath"], dtype=float)
#         self.hourly_sink_probs = np.array(season_block["sink"], dtype=float)
#         self.hourly_dishwasher_probs = np.array(season_block["dishwasher"], dtype=float)
#
#         # Ensure length = 24
#         for name, arr in [
#             ("shower", self.hourly_shower_probs),
#             ("bath", self.hourly_bath_probs),
#             ("sink", self.hourly_sink_probs),
#             ("dishwasher", self.hourly_dishwasher_probs),
#         ]:
#             if arr.shape[0] != 24:
#                 raise ValueError(f"Expected 24 values for '{name}', got {arr.shape[0]}.")
#
#         # Apply seasonal scaling factor
#         scale = SEASONAL_SCALING.get(self.season, 1.0)
#         self.hourly_shower_probs *= scale
#         self.hourly_bath_probs *= scale
#         self.hourly_sink_probs *= scale
#         self.hourly_dishwasher_probs *= scale
#
#     def _set_fixtures(self):
#         """
#         Define base parameters for each fixture:
#           - flow_mean (m³/sec), flow_std (m³/sec)
#           - dur_mean (sec), dur_std (sec)
#         Then apply any overrides and demographic adjustments.
#         """
#         # Base definitions (flow in m³/sec, then converted to L/min later)
#         self.fixtures = {
#             "sinkA":  {"flow_mean": 1/60/1000,  "flow_std": 1/60/1000,  "dur_mean": 60,   "dur_std": 30},
#             "sinkB":  {"flow_mean": 2/60/1000,  "flow_std": 1/60/1000,  "dur_mean": 60,   "dur_std": 30},
#             "shower": {"flow_mean": 8/60/1000,  "flow_std": 2/60/1000,  "dur_mean": 300,  "dur_std": 150},
#             "bath":   {"flow_mean": 10/60/1000, "flow_std": 1/60/1000,  "dur_mean": 300,  "dur_std": 120},
#             "dishwasher": {"flow_mean": 0.2/1000, "flow_std": 0.05/1000, "dur_mean": 3600, "dur_std": 900},
#         }
#
#         # 1) Override any base fixture params
#         for key, override in (self.fixture_params or {}).items():
#             if key in self.fixtures:
#                 self.fixtures[key].update(override)
#
#         # 2) Demographic adjustments
#         if self.demographics.get("has_children"):
#             # Children → slightly more sink and bath usage
#             self.fixtures["sinkA"]["flow_mean"] *= 1.2
#             self.fixtures["bath"]["dur_mean"] *= 1.2
#         if self.demographics.get("elderly"):
#             # Elderly → reduce all flows and durations
#             for fparams in self.fixtures.values():
#                 fparams["flow_mean"] *= 0.8
#                 fparams["dur_mean"] *= 0.8
#
#     def _minute_probabilities(self, hourly_probs):
#         """Repeat a length-24 array into a length-(24×60) array."""
#         return np.repeat(hourly_probs, 60)
#
#     def DHW_load_gen(self, return_in_kwh=False):
#         """
#         Generate minute-by-minute instantaneous flows (L/min) for:
#           - sinkA
#           - sinkB
#           - shower
#           - bath
#           - dishwasher
#
#         Steps:
#          1. Expand 24-hour probabilities to 1440-minute arrays.
#          2. For each minute, draw a Bernoulli trial with p = prob × (n_users/10).
#          3. If an event starts, sample duration and flow (m³/sec), convert to L/min, and fill.
#          4. After simulating the full day, calibrate so that total daily volume ≈ target_L_per_person × n_users.
#          5. Optionally, add an energy (kWh) variable assuming ΔT = 50 K.
#
#         Returns:
#           xarray.Dataset with variables:
#             sinkA, sinkB, shower, bath, dishwasher, total_L_per_min
#           (and total_kwh_per_min if return_in_kwh=True), coords={"time": minutes since midnight}.
#         """
#         # 1) Expand hourly to per-minute probabilities
#         prob_shower_min = self._minute_probabilities(self.hourly_shower_probs)
#         prob_bath_min = self._minute_probabilities(self.hourly_bath_probs)
#         prob_sink_min = self._minute_probabilities(self.hourly_sink_probs)
#         prob_dish_min = self._minute_probabilities(self.hourly_dishwasher_probs)
#
#         nstep = self.nday * 24 * 60
#         flows = {f: np.zeros(nstep) for f in self.fixtures}
#
#         # 2) Simulate each fixture minute by minute
#         for fixture, params in self.fixtures.items():
#             if fixture == "shower":
#                 base_probs = prob_shower_min
#             elif fixture == "bath":
#                 base_probs = prob_bath_min
#             elif fixture == "dishwasher":
#                 base_probs = prob_dish_min
#             else:
#                 base_probs = prob_sink_min  # sinkA or sinkB
#
#             for t in range(nstep):
#                 p_use = base_probs[t % len(base_probs)] * (self.n_users / 10.0)
#                 if random.random() < p_use:
#                     dur_sec = max(1, int(np.random.normal(params["dur_mean"], params["dur_std"])))
#                     flow_m3ps = max(0.0, np.random.normal(params["flow_mean"], params["flow_std"]))
#                     end = min(nstep, t + dur_sec)
#                     # Convert m³/sec → L/min: (m³/sec)*1000 = L/sec, *60 = L/min
#                     flow_L_per_min = flow_m3ps * 1000 * 60
#                     flows[fixture][t:end] += flow_L_per_min
#
#         # 3) Build xarray Dataset
#         time = np.arange(nstep)  # minutes since start
#         ds = xr.Dataset(
#             {
#                 "sinkA":  (["time"], flows["sinkA"]),
#                 "sinkB":  (["time"], flows["sinkB"]),
#                 "shower": (["time"], flows["shower"]),
#                 "bath":   (["time"], flows["bath"]),
#                 "dishwasher": (["time"], flows["dishwasher"]),
#             },
#             coords={"time": time},
#         )
#         ds["total_L_per_min"] = (
#             ds.sinkA + ds.sinkB + ds.shower + ds.bath + ds.dishwasher
#         )
#
#         # 4) CALIBRATION to target daily volume per person
#         total_volume_L = ds.total_L_per_min.sum().item()
#         desired_total = self.target_L_per_person * self.n_users * self.nday
#         if total_volume_L > 0:
#             scale_factor = desired_total / total_volume_L
#         else:
#             scale_factor = 0.0
#
#         for var in ["sinkA", "sinkB", "shower", "bath", "dishwasher", "total_L_per_min"]:
#             ds[var] = ds[var] * scale_factor
#
#         # 5) Optionally convert to energy (kWh) assuming ΔT = 50 K
#         if return_in_kwh:
#             # 1 L water = 1 kg; c_p ≈ 4.186 kJ/kg·K; ΔT = 50 K → 209.3 kJ = 0.05814 kWh per L
#             kwh_per_L = 4.186 * 50 / 3600.0
#             ds["total_kwh_per_min"] = ds.total_L_per_min * kwh_per_L
#
#         return ds
#
#
# # ------------------------------------------------------------------------------------
# # 4) Example usage and visualization
# # ------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Instantiate: 1 day, Belgium winter, 2 users with children, target 50 L/p·d
#     gen = StochasticDHWProfileFinal(
#         nday=1,
#         region="BE",
#         season="winter",
#         demographics={"has_children": True},
#         n_users=2,
#         seed=42,
#         target_L_per_person=50
#     )
#
#     # Generate flows (instantaneous L/min) and retrieve as xarray Dataset
#     ds = gen.DHW_load_gen(return_in_kwh=False)
#
#     # Plot total DHW flow (L/min) over the 1440 minutes
#     plt.figure(figsize=(10, 3))
#     ds.total_L_per_min.plot.line(x="time", color="tab:orange")
#     plt.title("Total DHW Flow (L/min) – Belgium, Winter, 2 Users")
#     plt.xlabel("Time (minutes from midnight)")
#     plt.ylabel("Flow (L/min)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     # Print daily total volume to confirm calibration
#     daily_total_L = ds.total_L_per_min.sum().item()
#     print(f"Total volume generated (L): {daily_total_L:.1f} (target was {2*50:.1f})")
#
#     # Plot per-fixture flows on a 2×3 grid
#     fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
#     fixtures = ["sinkA", "sinkB", "shower", "bath", "dishwasher"]
#     for ax, var in zip(axs.flatten(), fixtures + [None]):
#         if var is not None:
#             ax.plot(ds.time, ds[var].values, color="tab:orange", linewidth=1)
#             ax.set_title(f"{var} (L/min)")
#             ax.grid(True)
#         else:
#             ax.axis("off")  # last subplot unused
#
#     for ax in axs[-1]:
#         ax.set_xlabel("Time (minutes)")
#     plt.tight_layout()
#     plt.show()
import numpy as np
import random
import xarray as xr
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------
# 1) Household sampling utilities
# ------------------------------------------------------------------------------------
BELGIAN_HOUSEHOLD_TYPES = [
    ("one_person",               0.361, 1, {}),
    ("married_no_children",      0.182, 2, {}),
    ("married_with_children",    0.187, 4, {"has_children": True}),
    ("cohabiting_no_children",   0.066, 2, {}),
    ("cohabiting_with_children", 0.082, 3, {"has_children": True}),
    ("lone_parent",              0.099, 2, {"has_children": True}),
    ("other",                    0.022, 3, {}),
]

def sample_household_type():
    """
    Randomly pick a Belgian household type (name, average users, demographics dict),
    based on the probabilities in BELGIAN_HOUSEHOLD_TYPES.
    """
    types, probs, sizes, demos = zip(*BELGIAN_HOUSEHOLD_TYPES)
    probs = np.array(probs)
    probs = probs / probs.sum()
    idx = np.random.choice(len(types), p=probs)
    return types[idx], sizes[idx], demos[idx]


# ------------------------------------------------------------------------------------
# 2) Water‐use profile configuration: per‐region, per‐season, 24‐hour probability vectors
# ------------------------------------------------------------------------------------
# Each inner list has length 24 (one entry per hour).
# “dishwasher” is zero except in evening hours.
WATER_USE_PROFILES = {
    "BE": {
        "winter": {
            "shower": [
                0.013, 0.006, 0.004, 0.004, 0.009, 0.020, 0.050, 0.062,
                0.070, 0.065, 0.050, 0.040, 0.035, 0.030, 0.028, 0.028,
                0.032, 0.045, 0.050, 0.050, 0.045, 0.033, 0.025, 0.018
            ],
            # "bath": [
            #     0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.010, 0.015,
            #     0.020, 0.018, 0.015, 0.010, 0.008, 0.005, 0.005, 0.005,
            #     0.012, 0.020, 0.025, 0.030, 0.030, 0.025, 0.020, 0.010
            # ],
            "bath": [
                0.008, 0.004, 0.004, 0.004, 0.008, 0.019, 0.046, 0.058,
                0.066, 0.058, 0.046, 0.035, 0.031, 0.023, 0.023, 0.023,
                0.039, 0.046, 0.077, 0.100, 0.100, 0.077, 0.066, 0.039
                ],

            "sink": [
                0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.060,
                0.065, 0.060, 0.055, 0.050, 0.049, 0.047, 0.044, 0.045,
                0.050, 0.070, 0.075, 0.070, 0.060, 0.052, 0.042, 0.030
            ],
            "dishwasher": [
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.010, 0.020, 0.030, 0.020, 0.010, 0.000, 0.000
            ],
        },
        "spring": {
            "shower": [
                0.011, 0.005, 0.003, 0.003, 0.008, 0.018, 0.045, 0.055,
                0.060, 0.055, 0.045, 0.035, 0.030, 0.025, 0.023, 0.023,
                0.028, 0.040, 0.045, 0.045, 0.040, 0.030, 0.022, 0.015
            ],
            "bath": [
                0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.008, 0.012,
                0.015, 0.013, 0.012, 0.008, 0.007, 0.005, 0.005, 0.005,
                0.010, 0.018, 0.020, 0.025, 0.025, 0.020, 0.015, 0.008
            ],
            "sink": [
                0.014, 0.007, 0.005, 0.005, 0.007, 0.018, 0.044, 0.058,
                0.060, 0.058, 0.052, 0.048, 0.047, 0.045, 0.042, 0.043,
                0.046, 0.064, 0.070, 0.065, 0.055, 0.048, 0.040, 0.028
            ],
            "dishwasher": [
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.008, 0.015, 0.020, 0.015, 0.008, 0.000, 0.000
            ],
        },
        "summer": {
            "shower": [
                0.010, 0.004, 0.003, 0.003, 0.008, 0.017, 0.042, 0.052,
                0.058, 0.052, 0.042, 0.032, 0.028, 0.023, 0.020, 0.022,
                0.025, 0.035, 0.040, 0.040, 0.040, 0.035, 0.025, 0.017
            ],
            "bath": [
                0.001, 0.000, 0.000, 0.000, 0.001, 0.004, 0.007, 0.010,
                0.012, 0.010, 0.009, 0.006, 0.005, 0.004, 0.004, 0.004,
                0.008, 0.015, 0.018, 0.020, 0.020, 0.018, 0.015, 0.008
            ],
            "sink": [
                0.013, 0.007, 0.005, 0.005, 0.006, 0.018, 0.042, 0.056,
                0.058, 0.056, 0.048, 0.046, 0.045, 0.042, 0.040, 0.042,
                0.044, 0.060, 0.068, 0.064, 0.052, 0.046, 0.038, 0.026
            ],
            "dishwasher": [
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.006, 0.010, 0.015, 0.010, 0.006, 0.000, 0.000
            ],
        },
        "autumn": {
            "shower": [
                0.012, 0.006, 0.004, 0.004, 0.009, 0.019, 0.048, 0.060,
                0.065, 0.060, 0.050, 0.040, 0.036, 0.030, 0.028, 0.028,
                0.030, 0.043, 0.048, 0.048, 0.044, 0.032, 0.024, 0.017
            ],
            "bath": [
                0.002, 0.000, 0.000, 0.000, 0.001, 0.005, 0.009, 0.014,
                0.018, 0.016, 0.014, 0.010, 0.009, 0.006, 0.006, 0.006,
                0.014, 0.022, 0.028, 0.032, 0.032, 0.028, 0.022, 0.010
            ],
            "sink": [
                0.015, 0.008, 0.006, 0.006, 0.008, 0.020, 0.045, 0.058,
                0.062, 0.058, 0.053, 0.049, 0.048, 0.046, 0.044, 0.045,
                0.048, 0.068, 0.072, 0.068, 0.058, 0.050, 0.042, 0.030
            ],
            "dishwasher": [
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                0.000, 0.009, 0.015, 0.025, 0.015, 0.009, 0.000, 0.000
            ],
        },
    },
    "NL": {
        "winter":    {"shower": [0.012]*24, "bath": [0.008]*24, "sink": [0.014]*24, "dishwasher": [0.000]*24},
        "spring":    {"shower": [0.010]*24, "bath": [0.006]*24, "sink": [0.012]*24, "dishwasher": [0.000]*24},
        "summer":    {"shower": [0.009]*24, "bath": [0.005]*24, "sink": [0.011]*24, "dishwasher": [0.000]*24},
        "autumn":    {"shower": [0.011]*24, "bath": [0.007]*24, "sink": [0.013]*24, "dishwasher": [0.000]*24},
    },
}

# Seasonal scaling factors (relative to annual average)
SEASONAL_SCALING = {
    "winter": 1.12,   # +12 % DHW in winter
    "spring": 1.00,   # baseline
    "summer": 0.87,   # –13 % DHW in summer
    "autumn": 1.00,   # baseline
}


# ------------------------------------------------------------------------------------
# 3) Final DHW class with all enhancements
# ------------------------------------------------------------------------------------
class StochasticDHWProfileFinal:
    """
    Stochastic DHW profile generator with:
      • Region‐ and season‐aware start‐probabilities (shower, bath, sink, dishwasher)
      • Seasonal scaling (±12–13 %) of probabilities
      • Correct unit conversions → instantaneous L/min at each minute
      • Calibration to a target daily volume per person (default 50 L/p·d)
      • Optional energy output in kWh (ΔT = 50 K)
    """

    def __init__(
        self,
        nday=1,
        household_type=None,
        n_users=None,
        demographics=None,
        seed=None,
        fixture_params=None,
        region="BE",
        season="winter",
        target_L_per_person=50,
    ):
        # 1) Possibly sample a household type (if none given)
        if household_type is None:
            household_type, n_users, demographics = sample_household_type()

        self.household_type = household_type
        self.nday = nday
        self.n_users = n_users if n_users is not None else 1
        self.demographics = demographics or {}
        self.fixture_params = fixture_params or {}
        self.region = region
        self.season = season
        self.target_L_per_person = target_L_per_person

        # 2) Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 3) Load per‐hour probabilities and apply seasonal scaling
        self._load_time_of_day_probs()

        # 4) Build fixture definitions and apply overrides
        self._set_fixtures()

    def _load_time_of_day_probs(self):
        """Lookup and scale the 24‐hour probability lists for this region & season."""
        region_block = WATER_USE_PROFILES.get(self.region)
        if region_block is None:
            raise KeyError(f"No water‐use profile for region '{self.region}'.")

        season_block = region_block.get(self.season)
        if season_block is None:
            raise KeyError(f"No water‐use profile for season '{self.season}' in region '{self.region}'.")

        # Extract hourly probability vectors (as numpy arrays)
        self.hourly_shower_probs = np.array(season_block["shower"], dtype=float)
        self.hourly_bath_probs = np.array(season_block["bath"], dtype=float)
        self.hourly_sink_probs = np.array(season_block["sink"], dtype=float)
        self.hourly_dishwasher_probs = np.array(season_block["dishwasher"], dtype=float)

        # Ensure each is length 24
        for name, arr in [
            ("shower", self.hourly_shower_probs),
            ("bath", self.hourly_bath_probs),
            ("sink", self.hourly_sink_probs),
            ("dishwasher", self.hourly_dishwasher_probs),
        ]:
            if arr.shape[0] != 24:
                raise ValueError(f"Expected 24 values for '{name}', got {arr.shape[0]}.")

        # Apply the seasonal scaling factor
        scale = SEASONAL_SCALING.get(self.season, 1.0)
        self.hourly_shower_probs *= scale
        self.hourly_bath_probs *= scale
        self.hourly_sink_probs *= scale
        self.hourly_dishwasher_probs *= scale

    def _set_fixtures(self):
        """
        Define base parameters for each fixture:
          - flow_mean (m³/sec), flow_std (m³/sec)
          - dur_mean (sec), dur_std (sec)
        Then apply any overrides and demographic adjustments.
        """
        self.fixtures = {
            "sinkA":  {"flow_mean": 1/60/1000,  "flow_std": 1/60/1000,  "dur_mean": 60,   "dur_std": 30},
            "sinkB":  {"flow_mean": 2/60/1000,  "flow_std": 1/60/1000,  "dur_mean": 60,   "dur_std": 30},
            "shower": {"flow_mean": 8/60/1000,  "flow_std": 2/60/1000,  "dur_mean": 300,  "dur_std": 150},
            "bath":   {"flow_mean": 10/60/1000, "flow_std": 1/60/1000,  "dur_mean": 300,  "dur_std": 120},
            "dishwasher": {"flow_mean": 0.2/1000, "flow_std": 0.05/1000, "dur_mean": 3600, "dur_std": 900},
        }

        # 1) Override any base fixture params
        for key, override in (self.fixture_params or {}).items():
            if key in self.fixtures:
                self.fixtures[key].update(override)

        # 2) Demographic adjustments
        if self.demographics.get("has_children"):
            # Children → slightly more sink and bath usage
            self.fixtures["sinkA"]["flow_mean"] *= 1.2
            self.fixtures["bath"]["dur_mean"] *= 1.2
        if self.demographics.get("elderly"):
            # Elderly → reduce all flows and durations
            for fparams in self.fixtures.values():
                fparams["flow_mean"] *= 0.8
                fparams["dur_mean"] *= 0.8

    def _minute_probabilities(self, hourly_probs):
        """Repeat a length-24 array into a length-(24×60) array."""
        return np.repeat(hourly_probs, 60)

    def DHW_load_gen(self, return_in_kwh=False):
        """
        Generate minute-by-minute instantaneous flows (L/min) for:
          - sinkA
          - sinkB
          - shower
          - bath
          - dishwasher

        Steps:
          1. Expand 24-hour probabilities to 1440-minute arrays.
          2. For each minute, draw a Bernoulli trial with p = prob × (n_users/10).
          3. If an event starts, sample duration and flow (m³/sec), convert to L/min, and fill.
          4. After simulating the full day, report the raw total (L), then scale
             so that the daily total ≈ target_L_per_person × n_users.
          5. Optionally, add an energy (kWh) variable assuming ΔT = 50 K.

        Returns:
          (raw_ds, scaled_ds) as two xarray.Datasets with variables:
            sinkA, sinkB, shower, bath, dishwasher, total_L_per_min
          (and total_kwh_per_min if return_in_kwh=True), coords={"time": minutes since midnight}.
        """
        # 1) Expand each 24× array into 1440-minute arrays
        prob_shower_min = self._minute_probabilities(self.hourly_shower_probs)
        prob_bath_min = self._minute_probabilities(self.hourly_bath_probs)
        prob_sink_min = self._minute_probabilities(self.hourly_sink_probs)
        prob_dish_min = self._minute_probabilities(self.hourly_dishwasher_probs)

        nstep = self.nday * 24 * 60
        flows = {f: np.zeros(nstep) for f in self.fixtures}

        # 2) Simulate each fixture minute by minute
        for fixture, params in self.fixtures.items():
            if fixture == "shower":
                base_probs = prob_shower_min
            elif fixture == "bath":
                base_probs = prob_bath_min
            elif fixture == "dishwasher":
                base_probs = prob_dish_min
            else:
                base_probs = prob_sink_min  # sinkA or sinkB

            for t in range(nstep):
                p_use = base_probs[t % len(base_probs)] * (self.n_users / 10.0)
                if random.random() < p_use:
                    dur_sec = max(1, int(np.random.normal(params["dur_mean"], params["dur_std"])))
                    flow_m3ps = max(0.0, np.random.normal(params["flow_mean"], params["flow_std"]))
                    end = min(nstep, t + dur_sec)

                    # Convert m³/sec → L/min: (m³/sec)*1000 = L/sec, *60 = L/min
                    flow_L_per_min = flow_m3ps * 1000 * 60
                    flows[fixture][t:end] += flow_L_per_min

        # 3) Build an xarray Dataset with the raw (unscaled) flows
        time = np.arange(nstep)  # minutes since start
        raw_ds = xr.Dataset(
            {
                "sinkA":      (["time"], flows["sinkA"]),
                "sinkB":      (["time"], flows["sinkB"]),
                "shower":     (["time"], flows["shower"]),
                "bath":       (["time"], flows["bath"]),
                "dishwasher": (["time"], flows["dishwasher"]),
            },
            coords={"time": time},
        )
        raw_ds["total_L_per_min"] = (
            raw_ds.sinkA
            + raw_ds.sinkB
            + raw_ds.shower
            + raw_ds.bath
            + raw_ds.dishwasher
        )

        raw_total_volume = raw_ds.total_L_per_min.sum().item()

        # 4) CALIBRATION to target daily volume per person
        desired_total = self.target_L_per_person * self.n_users * self.nday

        if raw_total_volume > 0:
            scale_factor = desired_total / raw_total_volume
        else:
            scale_factor = 0.0

        # Create a copy for scaling
        scaled_ds = raw_ds.copy(deep=True)
        for var in ["sinkA", "sinkB", "shower", "bath", "dishwasher", "total_L_per_min"]:
            scaled_ds[var] = scaled_ds[var] * scale_factor

        # 5) Optionally convert to energy (kWh) assuming ΔT = 50 K
        if return_in_kwh:
            # 1 L water = 1 kg; c_p ≈ 4.186 kJ/kg·K; ΔT = 50 K → 4.186×50 = 209.3 kJ = 0.05814 kWh/L
            kwh_per_L = 4.186 * 50 / 3600.0
            scaled_ds["total_kwh_per_min"] = scaled_ds.total_L_per_min * kwh_per_L
            raw_ds["total_kwh_per_min"] = raw_ds.total_L_per_min * kwh_per_L

        return raw_ds, scaled_ds


# ------------------------------------------------------------------------------------
# 4) Example usage and visualization
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate: 1 day, Belgium winter, 2 users with children, target 50 L/p·d
    gen = StochasticDHWProfileFinal(
        nday=1,
        region="BE",
        season="winter",
        demographics={"has_children": True},
        n_users=2,
        seed=42,
        target_L_per_person=50
    )

    # Generate raw and scaled datasets
    raw_ds, scaled_ds = gen.DHW_load_gen(return_in_kwh=False)

    # 4a) Print raw vs. scaled daily volumes to confirm calibration
    raw_volume = raw_ds.total_L_per_min.sum().item()
    scaled_volume = scaled_ds.total_L_per_min.sum().item()
    print(f"Raw total volume (L):    {raw_volume:.1f}")
    print(f"Scaled total volume (L): {scaled_volume:.1f} (target was {2*50:.1f})")

    # 4b) Plot total DHW flow (L/min) – scaled
    plt.figure(figsize=(10, 3))
    scaled_ds.total_L_per_min.plot.line(x="time", color="tab:orange")
    plt.title("Scaled Total DHW Flow (L/min) – Belgium, Winter, 2 Users")
    plt.xlabel("Time (minutes from midnight)")
    plt.ylabel("Flow (L/min)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4c) Plot per-fixture instantaneous flows (scaled) on a 2×3 grid
    fixtures = ["sinkA", "sinkB", "shower", "bath", "dishwasher"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

    for ax, var in zip(axs.flatten(), fixtures + [None]):
        if var is not None:
            ax.plot(scaled_ds.time, scaled_ds[var].values, color="tab:orange", linewidth=1)
            ax.set_title(f"{var} (L/min)")
            ax.grid(True)
        else:
            ax.axis("off")  # last subplot unused

    for ax in axs[-1]:
        ax.set_xlabel("Time (minutes)")
    plt.tight_layout()
    plt.show()
