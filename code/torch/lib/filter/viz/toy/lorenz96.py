# SEABORN SETTINGS
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)


def plot_3d_attractor(x, var_buffer: int = 0):

    idx1 = var_buffer
    idx2 = idx1 + 1
    idx3 = idx2 + 1

    # Plot the first three variables
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(x[:, idx1], x[:, idx2], x[:, idx3], linewidth=4)

    ax.set_xlabel(f"$x({idx1})$")
    ax.set_ylabel(f"$x({idx2})$")
    ax.set_zlabel(f"$x({idx3})$")
    plt.show()

    return fig, ax


def plot_grid(model, sims):

    fig, ax = plt.subplots(figsize=(10, 5))

    extent = [sims.t[0], sims.t[-1], sims.x0.shape[0], 0]

    S = sims.x

    img = ax.imshow(S.T, extent=extent)

    ax.set(
        xlabel=f"Time Steps, $t$ ($\Delta t$ = {model.dt})",
        ylabel="Variables, $x^j$",
        aspect="auto",
    )
    plt.colorbar(img, label="State, $x^j$")

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_trajectories_horizontal(x):

    fig, ax = plt.subplots(figsize=(10, 5))

    pts = ax.plot(x.T, alpha=0.3, color="grey")

    ax.set(
        xlabel=f"Variables",
        ylabel="State",
        aspect="auto",
    )

    plt.legend([pts[0]], ["Trajectories"], loc="best", fontsize=12)
    # plt.legend()
    plt.show()

    return fig, ax


def plot_trajectories_vertical(x, t, dt):

    assert x.shape[0] == t.shape[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    pts = ax.plot(t, x, alpha=0.3, color="grey")

    ax.set(
        xlabel=f"Time Steps ($\Delta t$ = {dt})",
        ylabel="State",
        aspect="auto",
    )

    plt.legend([pts[0]], ["Variables"], loc="best", fontsize=12)
    # plt.legend()
    plt.show()
    return fig, ax
