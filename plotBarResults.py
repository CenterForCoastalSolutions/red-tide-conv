import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=15)

N = 3

ind = np.arange(N)  # the x locations for the groups
width = 0.08        # the width of the bars

fig, ax = plt.subplots(dpi=75, figsize=(16, 10))

bluesmap = matplotlib.cm.get_cmap('Blues')
cmap_inc = 20
cmap_start = 280

ours_means = (0.8292217839939031, 0.855885091302379, 0.6117195835185001)
ours_std = (0.03986438339760512, 0.06159873066094044, 0.07768124213147055)
rects1 = ax.bar(ind, ours_means, width, color='orange', yerr=ours_std, error_kw=dict(lw=5, capsize=5, capthick=3))

tomlinson_means = (0.7275876803591851, 0.7491224775839964, 0.4372834564066624)
tomlinson_std = (0.04491217960494535, 0.07169441499916836, 0.08077586804624638)
rects2 = ax.bar(ind + width, tomlinson_means, width, color=bluesmap(cmap_start-cmap_inc), yerr=tomlinson_std, error_kw=dict(lw=5, capsize=5, capthick=3))

hill_means = (0.7110265574453762, 0.7186896111560327, 0.43287092174283986)
hill_std = (0.10360681065154902, 0.13679662665360626, 0.13719175375619735)
rects3 = ax.bar(ind + 2*width, hill_means, width, color=bluesmap(cmap_start-2*cmap_inc), yerr=hill_std, error_kw=dict(lw=5, capsize=5, capthick=3))

soto_means = (0.7072569778830105, 0.7117943033758679, 0.420071670064445)
soto_std = (0.0374300355003253, 0.058216140093273305, 0.05673407379958459)
rects4 = ax.bar(ind + 3*width, soto_means, width, color=bluesmap(cmap_start-3*cmap_inc), yerr=soto_std, error_kw=dict(lw=5, capsize=5, capthick=3))

stumpf_means = (0.6900873008945209, 0.7837388478345155, 0.24644353618734183)
stumpf_std = (0.0924508453547038, 0.09317344819950331, 0.0705231517226841)
rects5 = ax.bar(ind + 4*width, stumpf_means, width, color=bluesmap(cmap_start-4*cmap_inc), yerr=stumpf_std, error_kw=dict(lw=5, capsize=5, capthick=3))

lou_means = (0.6635860267082292, 0.7609431491416181, 0.2022885568899619)
lou_std = (0.06623544753762546, 0.07329560864135676, 0.0539646588042963)
rects6 = ax.bar(ind + 5*width, lou_means, width, color=bluesmap(cmap_start-5*cmap_inc), yerr=lou_std, error_kw=dict(lw=5, capsize=5, capthick=3))

shehhi_means = (0.6243602778387818, 0.7574760200220632, 0.015306792070271946)
shehhi_std = (0.10109739868049906, 0.09084626445201446, 0.023974106005990398)
rects7 = ax.bar(ind + 6*width, shehhi_means, width, color=bluesmap(cmap_start-6*cmap_inc), yerr=shehhi_std, error_kw=dict(lw=5, capsize=5, capthick=3))

rbd_means = (0.6171334051394585, 0.6533600759728546, 0.21721425280707388)
rbd_std = (0.06206379439171269, 0.09789179414571049, 0.11459403847296608)
rects8 = ax.bar(ind + 7*width, rbd_means, width, color=bluesmap(cmap_start-7*cmap_inc), yerr=rbd_std, error_kw=dict(lw=5, capsize=5, capthick=3))

rbdkbbi_means = (0.6118068075997167, 0.535534931873137, 0.2911778440699078)
rbdkbbi_std = (0.06508595633824046, 0.11232605751185964, 0.07898858403631158)
rects9 = ax.bar(ind + 8*width, rbdkbbi_means, width, color=bluesmap(cmap_start-8*cmap_inc), yerr=rbdkbbi_std, error_kw=dict(lw=5, capsize=5, capthick=3))

cannizzaro2008_means = (0.580329461816058, 0.4993592611693137, 0.23492556867554093)
cannizzaro2008_std = (0.07084114773096284, 0.1177891937491023, 0.08594847883418086)
rects10 = ax.bar(ind + 9*width, cannizzaro2008_means, width, color=bluesmap(cmap_start-9*cmap_inc), yerr=cannizzaro2008_std, error_kw=dict(lw=5, capsize=5, capthick=3))

cannizzaro2009_means = (0.4861409248281056, 0.2967447033284528, 0.11768444622376087)
cannizzaro2009_std = (0.08717683594455493, 0.1337005456378108, 0.07101168852541051)
rects11 = ax.bar(ind + 10*width, cannizzaro2009_means, width, color=bluesmap(cmap_start-10*cmap_inc), yerr=cannizzaro2009_std, error_kw=dict(lw=5, capsize=5, capthick=3))

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(ind + 10*width / 2)
ax.set_xticklabels(('Accuracy', 'F1 Score', 'Kappa Coefficient'))

ax.set_ylim(0, 1)  # Add space for errorbar height
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0], rects8[0], rects9[0], rects10[0], rects11[0]), ('Spatio-temporal KNN+MLP', 'Tomlinson et al.', 'Hill et al.', 'Soto et al.', 'Stumpf et al.', 'Lou et al.', 'Shehhi et al.', 'Amin et al. (RBD)', 'Amin et al. (RBD+KBBI)', 'Cannizzaro et al. (2008)', 'Cannizzaro et al. (2009)'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """

    data_line, capline, barlinecols = rects.errorbar

    for err_segment, rect in zip(barlinecols[0].get_segments(), rects):
        height = err_segment[1][1]  # Use height of mean
        mean_height = rect.get_height()  # Use height of mean

        ax.text(rect.get_x() + rect.get_width() / 2, 
                1.05 * height,
                f'{mean_height:.2f}',
                ha='center', va='bottom', size=12)

#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)
#autolabel(rects5)
#autolabel(rects6)
#autolabel(rects7)
#autolabel(rects8)
#autolabel(rects9)
#autolabel(rects10)
#autolabel(rects11)

plt.savefig('bar_plot.png', bbox_inches='tight')