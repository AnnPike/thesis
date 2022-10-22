import matplotlib.pyplot as plt


def plot_4M_2A_precond(M_list, plot_over_iter_orig_bad, plot_over_iter_orig_good,
                               plot_over_iter_precond_bad, plot_over_iter_precond_good,
                       ylabel, degree, m, p, s, dict_of_cond):

    fig = plt.figure(figsize=(15, 10))
    # LSQR parameters
    iters = 40

    for i in range(4):
        M_random = M_list[i]


        plt.subplot(2, 2, i+1)
        l1 = plt.plot(plot_over_iter_orig_bad[i], c='b', label=f"cond(A) = {dict_of_cond[i]['orig bad']}")
        l2 = plt.plot(plot_over_iter_orig_good[i], c='y', label=f"cond(A) = {dict_of_cond[i]['orig good']}")
        l3 = plt.plot(plot_over_iter_precond_bad[i], c='purple', label=f"cond(A) = {dict_of_cond[i]['prec bad']}")
        l4 = plt.plot(plot_over_iter_precond_good[i], c='orange', label=f"cond(A) = {dict_of_cond[i]['prec good']}")

        plt.legend(loc='upper right')
        plt.xlabel('iterations')
        plt.yscale("log")
        plt.title(f'random state of M = {M_random}')
        plt.ylabel(ylabel)

    line_labels = [f'not scaled tensor (by 10^{degree})', 'scaled tensor', f'not scaled tensor (by 10^{degree})', 'scaled tensor']
    fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.8, 0.7))
    plt.suptitle(f'M prod LSQR, A shape is {m}, {p}, 2, s={s}')
    plt.tight_layout()
