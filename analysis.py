import matplotlib.pyplot as plt
import numpy as np
from collections import deque


K = 3.5
M = 0.2
L = 1.0
g = 9.8
theta0 = np.pi/4

theta_dot0 = 0.0
r0 = 1.5
r_dot0 = 0.0

dt = 0.01
T = 20
n_steps = int(T / dt)#time step

def get2ndR(r, theta, theta_dot):
    return r*theta_dot*theta_dot -K/M*(r-L) + g*np.cos(theta)

def get2ndTheta(r, r_dot, theta, theta_dot):
    return -(g*np.sin(theta) + 2*r_dot*theta_dot)/r


def calculate_energy(r, r_dot, theta, theta_dot):
    kinetic = 0.5 * M * (r_dot**2 + r**2 * theta_dot**2)

    spring_potential = 0.5 * K * (r - L)**2

    gravitational_potential = -M * g * r * np.cos(theta)

    total_energy = kinetic + spring_potential + gravitational_potential

    return total_energy


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

def useRK4(r, r_dot, theta, theta_dot):
    r_2nd = get2ndR(r, theta, theta_dot)
    theta_2nd = get2ndTheta(r, r_dot, theta, theta_dot)

    k1_r = dt*r_dot
    k1_r_dot = dt*r_2nd
    k1_theta = dt*theta_dot
    k1_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(r + 0.5*k1_r, theta + 0.5*k1_theta, theta_dot + 0.5*k1_theta_dot)
    theta_2nd = get2ndTheta(r + 0.5*k1_r, r_dot + 0.5*k1_r_dot, theta + 0.5*k1_theta, theta_dot + 0.5*k1_theta_dot)

    k2_r = dt*(r_dot + 0.5*k1_r_dot)
    k2_r_dot = dt*r_2nd
    k2_theta = dt*(theta_dot + 0.5*k1_theta_dot)
    k2_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(r + 0.5*k2_r, theta + 0.5*k2_theta, theta_dot + 0.5*k2_theta_dot)
    theta_2nd = get2ndTheta(r + 0.5*k2_r, r_dot + 0.5*k2_r_dot, theta + 0.5*k2_theta, theta_dot + 0.5*k2_theta_dot)

    k3_r = dt*(r_dot + 0.5*k2_r_dot)
    k3_r_dot = dt*r_2nd
    k3_theta = dt*(theta_dot + 0.5*k2_theta_dot)
    k3_theta_dot = dt*theta_2nd

    r_2nd = get2ndR(r + k3_r, theta + k3_theta, theta_dot + k3_theta_dot)
    theta_2nd = get2ndTheta(r + k3_r, r_dot + k3_r_dot, theta + k3_theta, theta_dot + k3_theta_dot)

    k4_r = dt*(r_dot + k3_r_dot)
    k4_r_dot = dt*r_2nd
    k4_theta = dt*(theta_dot + k3_theta_dot)
    k4_theta_dot = dt*theta_2nd

    r += (k1_r + 2*k2_r + 2*k3_r + k4_r)/6
    r_dot += (k1_r_dot + 2*k2_r_dot + 2*k3_r_dot + k4_r_dot)/6
    theta += (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)/6
    theta_dot += (k1_theta_dot + 2*k2_theta_dot + 2*k3_theta_dot + k4_theta_dot)/6

    return r, r_dot, theta, theta_dot

theta = theta0
theta_dot = theta_dot0
r = r0
r_dot = r_dot0
time_values = []
energy_values = []

fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-3, 0.5)
ax.set_aspect('equal')
trail, = ax.plot([], [], '-', color='tab:blue', lw=1, alpha=0.6)  # trace
spring, = ax.plot([], [], 'k-')
bob, = ax.plot([], [], 'ro', ms=10)
ax.plot(0, 0, 'ks')  # pivot

TRAIL_LEN = None                      # number of points kept; use None for the full path
xs = deque(maxlen=TRAIL_LEN)
ys = deque(maxlen=TRAIL_LEN)

for step in range(n_steps):
    current_time = step * dt

    current_energy = calculate_energy(
    r, r_dot, theta, theta_dot
)

    time_values.append(current_time)
    energy_values.append(current_energy)

    x, y = r*np.sin(theta), -r*np.cos(theta)
    xs.append(x); ys.append(y) 
    if False:
            trail.set_data(xs, ys)
            spring.set_data([0, x], [0, y])
            bob.set_data([x], [y])
            plt.pause(0.02)
    
    #using verlet integration to solve the ODEs
    #r, r_dot, theta, theta_dot = useVerlet(r, r_dot, theta, theta_dot)

    #using RK4 integration to solve the ODEs
    r, r_dot, theta, theta_dot = useRK4(r, r_dot, theta, theta_dot)

plt.close(fig)
initial_energy = calculate_energy(
    r0, r_dot0, theta0, theta_dot0
)

print("Initial Energy:", initial_energy)
plt.figure(figsize=(8, 5))

plt.plot(time_values, energy_values, color="blue")

plt.xlabel("Time (s)")
plt.ylabel("Total Mechanical Energy (J)")
plt.title("Energy Conservation Using verlet")

plt.grid(True)
plt.tight_layout()

plt.savefig("energy_vs_time.png", dpi=300)
plt.show()

initial_energy = energy_values[0]
final_energy = energy_values[-1]

percentage_error = (
    abs(final_energy - initial_energy)
    / abs(initial_energy)
) * 100

print("Initial Energy:", initial_energy)
print("Final Energy:", final_energy)
print("Percentage Energy Error:", percentage_error, "%")