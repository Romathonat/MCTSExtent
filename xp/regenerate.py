import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from xp.xp_main import annotate_bars
#sns.set()
#sns.palplot(sns.cubehelix_palette())


sns.set(rc={'figure.figsize': (8, 6.5)}, font_scale=1.35)
palette = 'inferno_r'
#YlOrRd
#gist_heat
plt.figure()
df = pd.read_pickle('./wracc_datasets/result')
ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df, palette=palette)
plt.savefig('./regenerate/iterations_boxplot.png')

plt.figure()
df = pd.read_pickle('./time_datasets/result')
ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df, palette=palette)
plt.savefig('./regenerate/time.png')

plt.figure()
df = pd.read_pickle('./improvement/mcts_bs_seqscout')
bx = sns.barplot(x='dataset', y='Improvement', data=df, palette=palette)
bx.set(yscale='symlog')
annotate_bars(bx)
plt.savefig('./regenerate/mcts_vs_seqscout.png')

plt.figure()
df = pd.read_pickle('./improvement/mcts_vs_beam')
bx = sns.barplot(x='dataset', y='Improvement', data=df, palette=palette)
bx.set(yscale='symlog')
annotate_bars(bx)
plt.savefig('./regenerate/mcts_vs_beam.png')

plt.figure()
df = pd.read_pickle('./improvement/mcts_vs_misere')
bx = sns.barplot(x='dataset', y='Improvement', data=df, palette=palette)
bx.set(yscale='symlog')
annotate_bars(bx)
plt.savefig('./regenerate/mcts_vs_misere.png')

plt.figure()
df = pd.read_pickle('./improvement/seqscout_vs_beam')
bx = sns.barplot(x='dataset', y='Improvement', data=df, palette=palette)
bx.set(yscale='symlog')
annotate_bars(bx)
plt.savefig('./regenerate/seqscout_vs_beam.png')

plt.figure()
df = pd.read_pickle('./improvement/seqscout_vs_misere')
bx = sns.barplot(x='dataset', y='Improvement', data=df, palette=palette)
bx.set(yscale='symlog')
annotate_bars(bx)
plt.savefig('./regenerate/seqscout_vs_misere.png')

for it in ['1000', '3000', '6000', '10000']:
    plt.figure()
    df = pd.read_pickle('./local_opti_increase/result_{}'.format(it))
    #ax = sns.barplot(x='dataset', y='WRAcc_opti', hue='Algorithm', data=df, ci=None, palette='muted')
    ax = sns.barplot(x='dataset', y='WRAcc_opti', hue='Algorithm', data=df, ci=None, palette=palette)
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df, ci=None, palette='Greys')
    #ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df, ci=None, palette='pastel')
    plt.legend(labels=['misere', 'beam', 'SeqScout', 'MCTSExtent'])
    plt.savefig('./regenerate/barplot_{}.png'.format(it))


plt.figure()
df = pd.read_pickle('./ground_truth/result')
red = ['#320656']
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=red)
plt.savefig('./regenerate/gt.png')

plt.figure()
df = pd.read_pickle('./theta/result')
ax = sns.lineplot(data=df, x='theta', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_theta.png')

plt.figure()
df = pd.read_pickle('./top_k/result')
ax = sns.lineplot(data=df, x='top_k', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_top_k.png')

plt.figure()
df = pd.read_pickle('./space_size/result')
ax = sns.lineplot(data=df, x='size', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
ax.set(xlabel='Length max', ylabel='WRAcc')
plt.savefig('./regenerate/quality_vs_size.png')

plt.figure()
df = pd.read_pickle('./number_iterations_optima/it_optima')
ax = sns.lineplot(data=df, x='iterations', y='cost', hue='dataset_name', style='dataset_name', dashes=False, markers=True, palette=palette)
plt.savefig('./regenerate/it_optima.png')

plt.figure()
df = pd.read_pickle('./lengths/result')
ax = sns.boxplot(x='dataset', y='Length', hue='Algorithm', data=df, palette=palette)
plt.savefig('./regenerate/boxplot.png')

# sns.set(rc={'figure.figsize': (8, 6.5)}, font_scale=1.6)
plt.figure()
df = pd.read_pickle('./iterations_ucb/result_splice')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_splice.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_context')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_context.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_jmlr')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_jmlr.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_promoters')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_promoters.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_rc')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_rc.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_sc2')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_sc2.png')

plt.figure()
df = pd.read_pickle('./iterations_ucb/result_skating')
ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm', style='Algorithm', markers=True, palette=palette)
plt.savefig('./regenerate/over_iter_skating.png')

plt.figure()
df = pd.read_pickle('./space_size/result')
ax = sns.lineplot(data=df, x='size', y='WRAcc', hue='Algorithm', markers=True, palette=palette)
ax.set(xlabel='Length max', ylabel='WRAcc')
plt.savefig('./regenerate/over_size.png')


