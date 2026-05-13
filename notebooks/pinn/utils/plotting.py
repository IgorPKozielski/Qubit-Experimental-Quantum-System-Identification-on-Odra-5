import numpy as np
import matplotlib.pyplot as plt


def plot_1q_trajectory(times, states, title="1-qubit Lindblad simulation"):

    states_np = states[:, 0, :, :]

    rho00 = np.real(states_np[:, 0, 0])
    rho11 = np.real(states_np[:, 1, 1])
    rho01 = np.abs(states_np[:, 0, 1])

    plt.figure(figsize=(8, 5))

    plt.plot(times, rho00, label="rho00")
    plt.plot(times, rho11, label="rho11")
    plt.plot(times, rho01, label="|rho01|")

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()