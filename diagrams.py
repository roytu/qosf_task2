
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from scipy.signal import savgol_filter

def plot_cost_histories():
    for M in [1, 10, 100, 1000]:
        costs = np.load(f"results/costs_{M}.npy")
        costs_smoothed = savgol_filter(costs, 51, 3)
    
        plt.plot(costs_smoothed, label=f"M={M}")
    
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_cost_histories()
