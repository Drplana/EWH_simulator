import numpy as np
import matplotlib.pyplot as plt

# Baseline power profile (from hysteresis control)
P_baseline = np.array([1, 2, 3, 3, 2])  # kW
timesteps = np.arange(1, 6)

# Shedding window: t3 and t4 (indices 2 and 3), shed 2 kW
P_shedded = P_baseline.copy()
for t in [2, 3]:  # t3 and t4
    P_shedded[t] = max(P_baseline[t] - 2, 0)

plt.plot(timesteps, P_baseline, label="Baseline (no shedding)")
plt.plot(timesteps, P_shedded, label="With shedding", marker="o")
plt.xlabel("Timestep")
plt.ylabel("Power (kW)")
plt.legend()
plt.show()