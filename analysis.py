
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import perf_counter

# ==========================================
# 1. PHYSICAL PARAMETERS
# ==========================================

K = 3.5
M = 0.2
L = 1.0
g = 9.8

theta0 = np.pi / 4
theta_dot0 = 0.0
r0 = 1.5
r_dot0 = 0.0

dt = 0.01
T = 20
n_steps = int(T / dt)

# Folder for saving graphs
results_folder = Path(__file__).resolve().parent / "results"
results_folder.mkdir(exist_ok=True)


# ==========================================
# 2. EQUATIONS OF MOTION
# ==========================================

def get2ndR(r, theta, theta_dot):
    return r*theta_dot*theta_dot - K/M*(r-L) + g*np.cos(theta)


def get2ndTheta(r, r_dot, theta, theta_dot):
    return -(g*np.sin(theta) + 2*r_dot*theta_dot)/r


# ==========================================
# 3. TOTAL MECHANICAL ENERGY
# ==========================================

def calculate_energy(r, r_dot, theta, theta_dot):

    kinetic = 0.5*M*(r_dot**2 + r**2*theta_dot**2)

    spring_potential = 0.5*K*(r-L)**2

    gravitational_potential = -M*g*r*np.cos(theta)

    total_energy = kinetic + spring_potential + gravitational_potential

    return total_energy


# ==========================================
# 4. VERLET-LIKE INTEGRATION
# ==========================================

def useVerlet(r, r_dot, theta, theta_dot):

    r_2nd = get2ndR(r, theta, theta_dot)
    theta_2nd = get2ndTheta(r, r_dot, theta, theta_dot)

    r += r_dot*dt + 0.5*r_2nd*dt*dt
    theta += theta_dot*dt + 0.5*theta_2nd*dt*dt

    r_dot += 0.5*r_2nd*dt
    theta_dot += 0.5*theta_2nd*dt

    r_2nd = get2ndR(r, theta, theta_dot)
    theta_2nd = get2ndTheta(r, r_dot, theta, theta_dot)

    r_dot += 0.5*r_2nd*dt
    theta_dot += 0.5*theta_2nd*dt

    return r, r_dot, theta, theta_dot


# ==========================================
# 5. RK4 INTEGRATION
# ==========================================

def useRK4(r, r_dot, theta, theta_dot):

    r_2nd = get2ndR(r, theta, theta_dot)
    theta_2nd = get2ndTheta(r, r_dot, theta, theta_dot)

    k1_r = dt*r_dot
    k1_r_dot = dt*r_2nd
    k1_theta = dt*theta_dot
    k1_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(
        r + 0.5*k1_r,
        theta + 0.5*k1_theta,
        theta_dot + 0.5*k1_theta_dot
    )
    theta_2nd = get2ndTheta(
        r + 0.5*k1_r,
        r_dot + 0.5*k1_r_dot,
        theta + 0.5*k1_theta,
        theta_dot + 0.5*k1_theta_dot
    )

    k2_r = dt*(r_dot + 0.5*k1_r_dot)
    k2_r_dot = dt*r_2nd
    k2_theta = dt*(theta_dot + 0.5*k1_theta_dot)
    k2_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(
        r + 0.5*k2_r,
        theta + 0.5*k2_theta,
        theta_dot + 0.5*k2_theta_dot
    )
    theta_2nd = get2ndTheta(
        r + 0.5*k2_r,
        r_dot + 0.5*k2_r_dot,
        theta + 0.5*k2_theta,
        theta_dot + 0.5*k2_theta_dot
    )

    k3_r = dt*(r_dot + 0.5*k2_r_dot)
    k3_r_dot = dt*r_2nd
    k3_theta = dt*(theta_dot + 0.5*k2_theta_dot)
    k3_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(
        r + k3_r,
        theta + k3_theta,
        theta_dot + k3_theta_dot
    )
    theta_2nd = get2ndTheta(
        r + k3_r,
        r_dot + k3_r_dot,
        theta + k3_theta,
        theta_dot + k3_theta_dot
    )

    k4_r = dt*(r_dot + k3_r_dot)
    k4_r_dot = dt*r_2nd
    k4_theta = dt*(theta_dot + k3_theta_dot)
    k4_theta_dot = dt*theta_2nd

    r += (k1_r + 2*k2_r + 2*k3_r + k4_r)/6
    r_dot += (k1_r_dot + 2*k2_r_dot + 2*k3_r_dot + k4_r_dot)/6
    theta += (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)/6
    theta_dot += (k1_theta_dot + 2*k2_theta_dot + 2*k3_theta_dot + k4_theta_dot)/6

    return r, r_dot, theta, theta_dot


# ==========================================
# 6. RUN SIMULATION
# ==========================================

def run_simulation(method):

    r = r0
    r_dot = r_dot0
    theta = theta0
    theta_dot = theta_dot0

    time_values = []
    energy_values = []
    radial_values = []

    start_time = perf_counter()

    for step in range(n_steps + 1):

        current_time = step * dt

        current_energy = calculate_energy(
            r, r_dot, theta, theta_dot
        )

        time_values.append(current_time)
        energy_values.append(current_energy)
        radial_values.append(r)

        if step < n_steps:
            r, r_dot, theta, theta_dot = method(
                r, r_dot, theta, theta_dot
            )

    execution_time = perf_counter() - start_time

    return (
        np.array(time_values),
        np.array(energy_values),
        np.array(radial_values),
        execution_time
    )


# ==========================================
# 7. RUN RK4 AND VERLET
# ==========================================

t_rk, E_rk, r_rk, runtime_rk = run_simulation(useRK4)

t_v, E_v, r_v, runtime_v = run_simulation(useVerlet)


# ==========================================
# 8. CALCULATE NUMERICAL ERROR
# ==========================================

initial_energy = calculate_energy(
    r0, r_dot0, theta0, theta_dot0
)

error_rk = (
    np.abs(E_rk - initial_energy)
    / abs(initial_energy)
) * 100

error_v = (
    np.abs(E_v - initial_energy)
    / abs(initial_energy)
) * 100


# ==========================================
# 9. FUNCTION TO SAVE GRAPHS
# ==========================================

def save_graph(filename):
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_folder / filename, dpi=300)
    plt.close()


# ==========================================
# 10. GRAPH 1: RK4 ENERGY CONSERVATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(t_rk, E_rk, color="blue")

plt.xlabel("Time (s)")
plt.ylabel("Total Mechanical Energy (J)")
plt.title("Energy Conservation Using RK4")

save_graph("energy_rk4.png")


# ==========================================
# 11. GRAPH 2: VERLET ENERGY CONSERVATION
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(t_v, E_v, color="orange")

plt.xlabel("Time (s)")
plt.ylabel("Total Mechanical Energy (J)")
plt.title("Energy Conservation Using Verlet-like Method")

save_graph("energy_verlet.png")


# ==========================================
# 12. GRAPH 3: NUMERICAL ERROR COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

# Exclude initial zero error from log plot
plt.semilogy(
    t_rk[1:],
    np.maximum(error_rk[1:], 1e-15),
    label="RK4"
)

plt.semilogy(
    t_v[1:],
    np.maximum(error_v[1:], 1e-15),
    label="Verlet-like"
)

plt.xlabel("Time (s)")
plt.ylabel("Relative Energy Error (%)")
plt.title("Numerical Energy Error Comparison")

plt.legend()

save_graph("energy_error_comparison.png")


# ==========================================
# 13. GRAPH 4: RADIAL MOTION COMPARISON
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(t_rk, r_rk, label="RK4")

plt.plot(
    t_v, r_v,
    linestyle="--",
    label="Verlet-like"
)

plt.xlabel("Time (s)")
plt.ylabel("Radial Distance (m)")
plt.title("Radial Motion Comparison")

plt.legend()

save_graph("radial_comparison.png")


# ==========================================
# 14. GRAPH 5: RUNTIME COMPARISON
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["RK4", "Verlet-like"],
    [runtime_rk, runtime_v],
    color=["blue", "orange"]
)

plt.ylabel("Execution Time (s)")
plt.title("Computational Runtime Comparison")

save_graph("runtime_comparison.png")


# ==========================================
# 15. PRINT NUMERICAL RESULTS
# ==========================================

print("\n========== CRITERION 4 RESULTS ==========")

print("\nInitial Energy:", initial_energy)

print("\nRK4:")
print("Final Energy:", E_rk[-1])
print("Final Percentage Error:", error_rk[-1], "%")
print("Maximum Percentage Error:", np.max(error_rk), "%")
print("Execution Time:", runtime_rk, "seconds")

print("\nVerlet-like:")
print("Final Energy:", E_v[-1])
print("Final Percentage Error:", error_v[-1], "%")
print("Maximum Percentage Error:", np.max(error_v), "%")
print("Execution Time:", runtime_v, "seconds")

print("\nAll graphs saved in:", results_folder)
