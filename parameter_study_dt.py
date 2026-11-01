
import matplotlib.pyplot as plt
from pathlib import Path

import analysis

# Same simulation duration
T = 20

# Different time steps
dt_values = [0.1, 0.05, 0.02, 0.01, 0.005]

errors = []

# Folder for saving graph
results_folder = Path(__file__).resolve().parent / "results"
results_folder.mkdir(exist_ok=True)

# Use the same initial energy
E0 = analysis.calculate_energy(
    analysis.r0,
    analysis.r_dot0,
    analysis.theta0,
    analysis.theta_dot0
)

for h in dt_values:

    # Change the time step used by your original RK4 code
    analysis.dt = h

    r = analysis.r0
    r_dot = analysis.r_dot0
    theta = analysis.theta0
    theta_dot = analysis.theta_dot0

    max_error = 0

    for i in range(round(T / h)):

        r, r_dot, theta, theta_dot = analysis.useRK4(
            r, r_dot, theta, theta_dot
        )

        E = analysis.calculate_energy(
            r, r_dot, theta, theta_dot
        )

        error = abs(E - E0) / abs(E0) * 100
        max_error = max(max_error, error)

    errors.append(max_error)

    print("dt =", h, "Maximum energy error =", max_error, "%")

# Plot time step vs maximum energy error
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, errors, "o-")
plt.xlabel("Time step dt (s)")
plt.ylabel("Maximum energy error (%)")
plt.title("Effect of Time Step on RK4 Accuracy")
plt.grid(True, which="both")
plt.tight_layout()

plt.savefig(results_folder / "dt_vs_error.png", dpi=300)
plt.show()
