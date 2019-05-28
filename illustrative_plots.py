import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# stylesheet_path = "https://github.com/Timothysit/normative_change_detect/blob/master/ts.mplstyle"
stylesheet_path = "ts.mplstyle"

def plot_epsilon_greedy(epsilon_list=[0.5], bias_list=[0.5]):
    plt.style.use(stylesheet_path)
    input = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots()
    ax.grid()
    for epsilon, bias in zip(epsilon_list, bias_list):
        output = input * (1 - epsilon) + epsilon * bias
        ax.plot(input, output, label=r"$\epsilon$ = %.2f, bias = %.2f" % (epsilon, bias))
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.set_xlabel(r"$p(\hat{a} = R \vert s) = p(s \vert x)$")
    ax.set_ylabel(r"$p(\hat{a} = R \vert s) = p(\hat{a} = R \vert s)(1 - \epsilon) + \epsilon p_\mathrm{bias}$")

    return fig, ax

def softmax_decision(beta_list=[0.5]):
    plt.style.use(stylesheet_path)
    fig, ax = plt.subplots()
    ax.grid()
    left_right_value_difference = np.linspace(-1, 1, 1000)
    for beta in beta_list:
        output = 1 / (1 + np.exp(-beta * left_right_value_difference))
        ax.plot(left_right_value_difference, output, label=r"$\beta$ = %.2f" % (beta))

    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend()

    ax.set_ylabel(r"$p(\hat{a} = R \vert s) = p(s \vert x)$")
    ax.set_xlabel(r"$Q(R) - Q(L)$")


    return fig, ax

def softmax_decision_with_value(beta_list=[0.5], alpha_list=[0.5], x_axis="posterior", plot_vline=False):
    """
    plots softmax decision rule where the alpha is the false alarm cost relative to the reward
    this impelemntation assuems that the value of not licking is zero
    :param beta_list:
    :param alpha_list:
    :return:
    """
    plt.style.use(stylesheet_path)
    fig, ax = plt.subplots()
    ax.grid()

    p_change = np.linspace(0, 1, 1000)
    for beta, alpha in zip(beta_list, alpha_list):
        q_lick = p_change * (1 - alpha) + alpha
        pi = 1 / (1 + np.exp(-beta * q_lick))
        if x_axis == "posterior":
            ax.plot(p_change, pi, label=r"$\alpha$ = %.2f, $\beta$ = %.2f" % (alpha, beta))
            if plot_vline is True:
                ax.axvline(1 / (1 - 1/alpha), ls="--", linewidth=0.7, c='gray')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel(r"$p(\mathrm{change})$")
        elif x_axis == "value":
            ax.plot(q_lick, pi, label=r"$\alpha$ = %.2f, $\beta$ = %.2f" % (alpha, beta))
            ax.set_xlabel(r"$Q_\mathrm{lick}$")

    # midpoint
    if (x_axis == "posterior") and (plot_vline is True):
        ax.axvline(1 / (1 - 1/alpha), ls="--", linewidth=0.7, c='gray', label=r"$\frac{1}{1-\alpha^{-1}}$")

    ax.set_ylabel(r"$p(\mathrm{lick})$")
    ax.legend()

    return fig, ax

def example_signal(tau=50, signal_length=300, signal_var=0.1, baseline=1.0, change_amplitude=1.25):
    """
    Plots example signal trace.
    :param tau:
    :param signal_var:
    :param baseline:
    :param change_amplitude:
    :return:
    """
    plt.style.use(stylesheet_path)
    fig, ax = plt.subplots()
    ax.grid()

    baseline_signal = np.random.normal(loc=baseline, scale=signal_var, size=tau)
    change_signal = np.random.normal(loc=change_amplitude, scale=signal_var, size=signal_length-tau)
    ax.plot(np.arange(0, signal_length), np.concatenate([baseline_signal, change_signal]))

    ax.set_ylabel("Speed")
    ax.set_xlabel("Time (frames)")

    return fig, ax

def normal_dist(mean=0, var=1, vert_loc=None):
    """
    Plots example normal distribution
    :param mean:
    :param var:
    :return:
    """
    fig, ax = plt.subplots()

    x = np.linspace(-3, 3, 1000)
    p_x = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))
    ax.plot(x, p_x)

    ax.set_xlabel(r"$x_t$")
    ax.set_ylabel(r"$p(x_t \vert z)$")

    ax.grid()

    if vert_loc is not None:
        p_vert_loc = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(vert_loc - mean) ** 2 / (2 * var))
        ax.scatter(vert_loc, p_vert_loc, color="r", s=20)
        ax.vlines(vert_loc, ymin=0, ymax=p_vert_loc, linestyle="--", color="r")
        ax.hlines(p_vert_loc, xmin=-3, xmax=vert_loc, linestyle="--", color="r")

    return fig, ax

def change_time_dist(early_block_time=[3, 8], late_block_time=[10.5, 15.5], early_block_prop=0.9,
                     late_block_prop=0.1, total_sample_size=10000, dist_type="uniform"):
    """
    Plot distribution of change times (idealised)
    :return:
    """
    plt.style.use(stylesheet_path)
    fig, ax = plt.subplots()

    if dist_type == "uniform":
        early_change_times = np.random.uniform(low=early_block_time[0], high=early_block_time[1],
                                               size=int(total_sample_size * early_block_prop))
        late_change_times = np.random.uniform(low=late_block_time[0], high=late_block_time[1],
                                              size=int(total_sample_size * late_block_prop))
    elif dist_type == "exponential":
        pass



    sns.distplot(np.concatenate([early_change_times, late_change_times]), kde=True, ax=ax)

    ax.set_ylabel("Kernel density estimate")
    ax.set_xlabel("Change time (seconds)")

    return fig, ax

def main():
    fig_folder = "/home/timothysit/Dropbox/notes/Projects/second_rotation_project/normative_model/figures/illustrations"
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    # epsilon greedy with varying beta
    figsavepath = os.path.join(fig_folder, "epsilon_greedy_vary_beta")
    fig, ax = plot_epsilon_greedy(epsilon_list=[0.5, 0.5, 0.5], bias_list=[1.0, 0.5, 0.0])
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    # epsilon greedy with varying bias
    figsavepath = os.path.join(fig_folder, "epsilon_greedy_vary_bias")
    fig, ax = plot_epsilon_greedy(epsilon_list=[0.0, 0.5, 1.0], bias_list=[0.5, 0.5, 0.5])
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    # softmax with varying beta
    figsavepath = os.path.join(fig_folder, "softmax_vary_beta")
    fig, ax = softmax_decision(beta_list=[1, 10, 100])
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    figsavepath = os.path.join(fig_folder, "softmax_decision_with_value_vary_alpha_v3")
    fig, ax = softmax_decision_with_value(beta_list=[5, 5, 5], alpha_list=[-1, -2, -3], x_axis="posterior")
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    figsavepath = os.path.join(fig_folder, "softmax_decision_with_value_vary_beta_value_xaxis")
    fig, ax = softmax_decision_with_value(beta_list=[1, 5, 10], alpha_list=[-2, -2, -2], x_axis="value")
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    # softmax with a mixure of things
    figsavepath = os.path.join(fig_folder, "softmax_decision_with_value_vary_both")
    fig, ax = softmax_decision_with_value(beta_list=[1, 1, 10, 10], alpha_list=[-2, -5, -2, -5], x_axis="posterior",
                                          plot_vline=False)
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)


    # normal distribution
    figsavepath = os.path.join(fig_folder, "normal_dist_mean_0_var_1")
    fig, ax = normal_dist(mean=0, var=1)
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath)

    # normal distribution with estimate
    figsavepath = os.path.join(fig_folder, "normal_dist_mean_0_var_1_vertline")
    fig, ax = normal_dist(mean=0, var=1, vert_loc=1.0)
    fig.set_size_inches(4, 4)
    fig.savefig(figsavepath, dpi=600)  # format="svg"

    # Example signal
    figsavepath = os.path.join(fig_folder, "change_signal")
    fig, ax = example_signal(tau=50, signal_length=300, signal_var=0.1, baseline=1.0, change_amplitude=1.25)
    fig.set_size_inches(8, 4)
    fig.savefig(figsavepath, dpi=300)

    # Change time distribution
    figsavepath = os.path.join(fig_folder, "change_time_dist")
    fig, ax = change_time_dist(early_block_time=[3, 8], late_block_time=[10.5, 15.5], early_block_prop=0.9,
                     late_block_prop=0.1, total_sample_size=10000)
    fig.set_size_inches(8, 4)
    fig.savefig(figsavepath, dpi=300)



if __name__ == "__main__":
    main()