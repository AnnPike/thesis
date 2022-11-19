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
    sigma_list = dict_of_lines[0].keys()
    line_styles = dict(zip(sigma_list, ['solid', 'dashed', 'dotted']))
    for i in range(4):
        M_random = M_list[i]

        plt.subplot(2, 2, i+1)
        for sigma in sigma_list:
            l1 = plt.plot(dict_of_lines[i][sigma]['orig good'], c='y', label=f"sigma={sigma}", ls=line_styles[sigma])
            l2 = plt.plot(dict_of_lines[i][sigma]['orig bad'], c='b', label=f"sigma={sigma}", ls=line_styles[sigma])
            l3 = plt.plot(dict_of_lines[i][sigma]['prec good'], c='orange', label=f"sigma={sigma}", ls=line_styles[sigma])
            l4 = plt.plot(dict_of_lines[i][sigma]['prec bad'], c='purple', label=f"sigma={sigma}", ls=line_styles[sigma])


            plt.plot(dict_of_lines[i][sigma]['orig good'], '^', c='y', markevery=5, markersize=10, mfc='none')
            plt.plot(dict_of_lines[i][sigma]['orig bad'], 'x', c='b', markevery=5, markersize=10, mfc='none')
            plt.plot(dict_of_lines[i][sigma]['prec good'], 's', c='orange', markevery=5, markersize=10, mfc='none')
            plt.plot(dict_of_lines[i][sigma]['prec bad'], 'o', c='purple', markevery=5, markersize=10, mfc='none')



        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('iterations', fontsize=13)
        plt.yscale("log")
        plt.title(f'random state of M = {M_random}', fontsize=15)
        plt.ylabel(dict_labels[plot_what], fontsize=13)
        plt.grid(visible=True, which='major', axis='both', ls=':', alpha=0.5)
    line_labels = ['scaled tensor', f'not scaled tensor (by $10^{degree}$)', 'BLENDENPIK scaled tensor', f'BLENDENPIK not scaled tensor (by $10^{degree})$']
    fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.9, 0.6), ncol=2, fontsize=15)
    plt.suptitle(f'M prod LSQR, A shape is {m}, {p}, 2, s={s}', fontsize=17)
    plt.tight_layout()
