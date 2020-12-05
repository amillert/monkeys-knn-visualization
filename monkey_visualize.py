import matplotlib.pyplot as plt

import os


def scatter_plot(X: list, Y: list, labels: list):
    assert len(X) == len(Y) == len(labels), (
        f"Lengths mismatch: X -> {len(X)}, y -> {len(Y)}, labels -> {len(labels)}")

    colors = iter(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    species2color = {s: next(colors) for s in set(labels)}

    introduced_species = set()
    for x, y, l in zip(X, Y, labels):
        if l not in introduced_species:
            introduced_species.add(l)
            plt.scatter(x, y, label=l, color=species2color[l], alpha=0.68)
        else:
            plt.scatter(x, y, color=species2color[l], alpha=0.68)

    plt.legend()
    plt.show()
    path = os.path.join(os.path.join(os.curdir, "figs"), f"fig_{len(os.listdir('./figs'))}")
    plt.savefig(path)
    print(f"Figured saved in {os.path.abspath(path)} path")
