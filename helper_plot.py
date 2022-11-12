import matplotlib.pyplot as plt


dict_labels = {'residual':'$\Vert\overline{R}\Vert_F$',
            'error':'$\Vert\overline{E}\Vert_{A_s}$',
            'normalized residual':'$\Vert A^T\cdot\overline{R}\Vert_F/\Vert A\Vert_F\Vert \overline{R}\Vert_F$',
            'normalized error':'$\Vert\overline{E}\Vert_{A_s}/\Vert\overline{E}_0\Vert_{A_s}$'}

def plot_4M_2A_precond(M_list, dict_of_lines,
                               plot_what, degree, m, p, s, dict_of_cond):

    fig = plt.figure(figsize=(15, 10))
    # LSQR parameters
    iters = 40

    for i in range(4):
        M_random = M_list[i]


        plt.subplot(2, 2, i+1)
        l1 = plt.plot(dict_of_lines[i]['orig good'], c='y', label=f"cond(A) = {dict_of_cond[i]['orig good']: .0e}")
        l2 = plt.plot(dict_of_lines[i]['orig bad'], c='b', label=f"cond(A) = {dict_of_cond[i]['orig bad']: .0e}")
        l3 = plt.plot(dict_of_lines[i]['prec good'], c='orange', label=f"cond(A) = {dict_of_cond[i]['prec good']: .0e}")
        l4 = plt.plot(dict_of_lines[i]['prec bad'], c='purple', label=f"cond(A) = {dict_of_cond[i]['prec bad']: .0e}")

        plt.legend(loc='upper right')
        plt.xlabel('iterations')
        plt.yscale("log")
        plt.title(f'random state of M = {M_random}')
        plt.ylabel(dict_labels[plot_what])

    line_labels = ['scaled tensor', f'not scaled tensor (by $10^{degree}$)', 'BLENDENPIK scaled tensor', f'BLENDENPIK not scaled tensor (by $10^{degree})$']
    fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.9, 0.7), ncol=2, fontsize=12)
    plt.suptitle(f'M prod LSQR, A shape is {m}, {p}, 2, s={s}')
    plt.tight_layout()
