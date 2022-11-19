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
        l1 = plt.plot(dict_of_lines[i]['orig good'], c='y', label=f"cond(A) = {dict_of_cond[i]['orig good']: .0e}", lw=2)
        l2 = plt.plot(dict_of_lines[i]['orig bad'], c='b', label=f"cond(A) = {dict_of_cond[i]['orig bad']: .0e}", lw=2)
        l3 = plt.plot(dict_of_lines[i]['prec good'], c='orange', label=f"cond(A) = {dict_of_cond[i]['prec good']: .0e}", lw=2)
        l4 = plt.plot(dict_of_lines[i]['prec bad'], c='purple', label=f"cond(A) = {dict_of_cond[i]['prec bad']: .0e}", lw=2)

        plt.plot(dict_of_lines[i]['orig good'], '^', c='y', markevery=5, markersize=10, mfc='none')
        plt.plot(dict_of_lines[i]['orig bad'], 'x', c='b', markevery=5, markersize=10, mfc='none')
        plt.plot(dict_of_lines[i]['prec good'], 's', c='orange', markevery=5, markersize=10, mfc='none')
        plt.plot(dict_of_lines[i]['prec bad'], 'o', c='purple', markevery=5, markersize=10, mfc='none')

        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('iterations', fontsize=13)
        plt.yscale("log")
        plt.title(f'random state of M = {M_random}', fontsize=13)
        plt.ylabel(dict_labels[plot_what], fontsize=13)
        plt.grid(visible=True, which='major', axis='both', ls=':', alpha=0.5)

    line_labels = ['scaled tensor', f'not scaled tensor (by $10^{degree}$)', 'BLENDENPIK scaled tensor', f'BLENDENPIK not scaled tensor (by $10^{degree})$']
    fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.9, 0.6), ncol=2, fontsize=15)
    plt.suptitle(f'M prod LSQR, A shape is {m}, {p}, 2, s={s}', fontsize=17)

    plt.tight_layout()
