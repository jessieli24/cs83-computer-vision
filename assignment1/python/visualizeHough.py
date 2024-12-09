import numpy as np

import matplotlib.pyplot as plt

def visualizeHough(points):
    """
        For Problem 1, Part 4. Graphs the Hough transforation of each point
        for θ ∈ [0, π]. 
        
        Parameters:
            points: points to apply Hough transform to

        Returns:
            fig: matplotlib figure
            ax: matplotlib axis

    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True)

    theta = np.linspace(0, np.pi, 100)

    for x, y in points:
        ax.plot(theta, x * np.cos(theta) + y * np.sin(theta), label=fr"$\rho = {x} \cos\theta + {y} \sin\theta$")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\rho$")

    x_labels = [0, r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    ax.set_xticks(np.arange(0, 1.1*np.pi, np.pi/4), labels=x_labels)

    ax.legend()
    return fig, ax

fig, ax = visualizeHough(np.array([[10, 10], [20, 20], [30, 30]]))

# Intersection
ax.plot(3*np.pi/4, 0, 'ko')
ax.annotate(r"$\rho = 0$" + "\n" +  r"$\theta = 3\pi/4$", (3*np.pi/4, 4))

fig.savefig("../results/problem1part4.png")
