from multiprocessing.pool import Pool

import sys
import pathlib

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from competitors.beam_search import beam_search
from competitors.misere import misere

from general.reader import read_data, read_jmlr, read_data_sc2, read_data_kosarak
from general.utils import average_results, reduce_k_length, k_length, extract_items

from general.conf import TIME_BUDGET_XP, TOP_K, THETA
from general.priorityset import PrioritySet
from seqscout.seq_scout import seq_scout, optimize_pattern
import seqscout.global_var
from mctsextent.main import launch_mcts

sys.setrecursionlimit(500000)

# third element: enable_i
datasets = [
    (read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data'), '+', False),
    (read_data_kosarak(pathlib.Path(__file__).parent.parent / 'data/context.data'), '4', False),
    (read_data(pathlib.Path(__file__).parent.parent / 'data/splice.data'), 'EI', False),
    (read_data_sc2(pathlib.Path(__file__).parent.parent / 'data/sequences-TZ-45.txt')[:5000], '1', True),
    (read_data_kosarak(pathlib.Path(__file__).parent.parent / 'data/skating.data'), '1', False),
    (read_jmlr('svm', pathlib.Path(__file__).parent.parent / 'data/jmlr/jmlr'), '+', False),
    (read_data_kosarak('../data/figures_rc.dat'), '3', True)
]

datasets_names = ['promoters', 'context', 'splice', 'sc2', 'skating', 'jmlr', 'rc']

SHOW = False


def data_add_generic(data, **kwargs):
    for key, value in kwargs.items():
        data[key].append(value)


def barplot_dataset_iterations():
    pool = Pool(processes=5, maxtasksperchild=1)
    xp_repeat = 5

    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': TIME_BUDGET_XP})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})
            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_misere) < TOP_K:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < TOP_K:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print(
                    "Too few example on seqscout on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))
            if len(results_mcts) < TOP_K:
                print(
                    "Too few example on mctsextend on dataset {}: {} results".format(datasets_names[i],
                                                                                     len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), dataset=datasets_names[i],
                             Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), dataset=datasets_names[i],
                             Algorithm='SeqScout')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts)), dataset=datasets_names[i],
                             Algorithm='MCTSExtent')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)

    plt.savefig('./wracc_datasets/iterations_boxplot.png')
    df.to_pickle('./wracc_datasets/result')

    if SHOW:
        plt.show()


def barplot_dataset_time():
    pool = Pool(processes=5, maxtasksperchild=1)
    xp_repeat = 5
    time_budget = 60
    iteration_budget = 2 ** 30
    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': time_budget, 'iterations_limit': iteration_budget})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': time_budget, 'iterations_limit': iteration_budget})
            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': time_budget,
                                                'iterations_limit': iteration_budget})
            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': time_budget, 'iterations_limit': iteration_budget})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_misere) < TOP_K:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < TOP_K:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print(
                    "Too few example on seqscout on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))
            if len(results_mcts) < TOP_K:
                print(
                    "Too few example on mctsextend on dataset {}: {} results".format(datasets_names[i],
                                                                                     len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), dataset=datasets_names[i],
                             Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), dataset=datasets_names[i],
                             Algorithm='SeqScout')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts)), dataset=datasets_names[i],
                             Algorithm='MCTSExtent')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)

    plt.savefig('./time_datasets/time.png')
    df.to_pickle('./time_datasets/result')

    if SHOW:
        plt.show()


def compute_improvement(result1, result2):
    return (result1 - result2) / result2 * 100


def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate('{}%'.format(format(p.get_height(), '.2f')), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')


def barplot_dataset_improvement_iterations():
    pool = Pool(processes=5, maxtasksperchild=1)
    xp_repeat = 5

    seqscout_vs_beam = {'Improvement': [], 'dataset': []}
    seqscout_vs_misere = {'Improvement': [], 'dataset': []}
    mcts_vs_misere = {'Improvement': [], 'dataset': []}
    mcts_vs_beam = {'Improvement': [], 'dataset': []}
    mcts_vs_seqscout = {'Improvement': [], 'dataset': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP})

            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': TIME_BUDGET_XP})
            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})
            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_misere) < TOP_K:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < TOP_K:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print(
                    "Too few example on seqscout on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))
            if len(results_mcts) < TOP_K:
                print(
                    "Too few example on mctsextend on dataset {}: {} results".format(datasets_names[i],
                                                                                     len(results_mcts)))

            data_add_generic(seqscout_vs_misere, Improvement=compute_improvement(average_results(result_ucb_opti),
                                                                                 average_results(results_misere)),
                             dataset=datasets_names[i])

            data_add_generic(seqscout_vs_beam, Improvement=compute_improvement(average_results(result_ucb_opti),
                                                                               average_results(results_beam)),
                             dataset=datasets_names[i])
            data_add_generic(mcts_vs_misere, Improvement=compute_improvement(average_results(results_mcts),
                                                                             average_results(results_misere)),
                             dataset=datasets_names[i])
            data_add_generic(mcts_vs_beam, Improvement=compute_improvement(average_results(results_mcts),
                                                                           average_results(results_beam)),
                             dataset=datasets_names[i])
            data_add_generic(mcts_vs_seqscout, Improvement=compute_improvement(average_results(results_mcts),
                                                                               average_results(result_ucb_opti)),
                             dataset=datasets_names[i])

    sns.set(rc={'figure.figsize': (8, 6.5)})
    plt.clf()

    df = pd.DataFrame(data=seqscout_vs_misere)
    ax = sns.barplot(x='dataset', y='Improvement', data=df)
    ax.set(yscale='symlog')
    annotate_bars(ax)
    plt.savefig('./improvement/seqscout_vs_misere.png')
    df.to_pickle('./improvement/seqscout_vs_misere')
    plt.clf()

    df = pd.DataFrame(data=seqscout_vs_beam)
    bx = sns.barplot(x='dataset', y='Improvement', data=df)
    bx.set(yscale='symlog')
    annotate_bars(bx)
    plt.savefig('./improvement/seqscout_vs_beam.png')
    df.to_pickle('./improvement/seqscout_vs_beam')
    plt.clf()

    df = pd.DataFrame(data=mcts_vs_misere)
    cx = sns.barplot(x='dataset', y='Improvement', data=df)
    cx.set(yscale='symlog')
    annotate_bars(cx)
    plt.savefig('./improvement/mcts_vs_misere.png')
    df.to_pickle('./improvement/mcts_vs_misere')
    plt.clf()

    df = pd.DataFrame(data=mcts_vs_beam)
    dx = sns.barplot(x='dataset', y='Improvement', data=df)
    dx.set(yscale='symlog')
    annotate_bars(dx)
    plt.savefig('./improvement/mcts_vs_beam.png')
    df.to_pickle('./improvement/mcts_vs_beam')
    plt.clf()

    df = pd.DataFrame(data=mcts_vs_seqscout)
    ex = sns.barplot(x='dataset', y='Improvement', data=df)
    ex.set(yscale='symlog')
    annotate_bars(ex)
    plt.savefig('./improvement/mcts_vs_seqscout.png')
    df.to_pickle('./improvement/mcts_bs_seqscout')
    plt.clf()

    if SHOW:
        plt.show()


def show_quality_over_iterations_ucb(number_dataset):
    data, target, enable_i = datasets[number_dataset]

    # if we want to average
    nb_launched = 5
    pool = Pool(processes=3, maxtasksperchild=1)

    iterations_limit = 50
    iterations_step = 1000

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}

    for i in range(12):
        print('Iteration: {}'.format(i))

        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP, 'iterations_limit': iterations_limit})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                             'iterations_limit': iterations_limit})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'iterations_limit': iterations_limit})

            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP,
                                             'iterations_limit': iterations_limit})

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere.get())),
                             iterations=iterations_limit,
                             Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam.get())), iterations=iterations_limit,
                             Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti.get())),
                             iterations=iterations_limit,
                             Algorithm='SeqScout')

            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts.get())),
                             iterations=iterations_limit,
                             Algorithm='MCTSExtent')

        iterations_limit += iterations_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')

    plt.savefig('./iterations_ucb/over_iterations{}.png'.format(datasets_names[number_dataset]))
    df.to_pickle('./iterations_ucb/result_{}'.format(datasets_names[number_dataset]))

    if SHOW:
        plt.show()


def compare_ground_truth():
    data = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    data = reduce_k_length(10, data)

    target = '1'
    enable_i = True

    # if we want to average
    nb_launched = 5
    pool = Pool(processes=3, maxtasksperchild=1)

    iterations_limit = 50
    iteration_step = 1000

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}

    # found with exaustive search
    ground_truth = 0.008893952000000009

    for i in range(10):
        print('Iteration: {}'.format(i))

        for i in range(nb_launched):
            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'iterations_limit': iterations_limit})

            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti.get())) / ground_truth,
                             iterations=iterations_limit, Algorithm='SeqScout')

        iterations_limit += iteration_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})
    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')
    ax.set(xlabel='iterations', ylabel='WRAcc')
    plt.savefig('./ground_truth/gt.png')
    df.to_pickle('./ground_truth/result')

    if SHOW:
        plt.show()


def naive_vs_bitset_seqscout():
    time_xp = 10

    for i, (data, target, enable_i) in enumerate(datasets[-2:]):
        seq_scout(data, target, time_budget=time_xp, enable_i=enable_i, vertical=False, iterations_limit=2 ** 30)
        seq_scout(data, target, time_budget=time_xp, enable_i=enable_i, vertical=True, iterations_limit=2 ** 30)
    # we need to look at the console output


def other_measures():
    pool = Pool(processes=5, maxtasksperchild=1)
    xp_repeat = 5
    nb_iterations = 10000

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        for measure in ['Informedness', 'F1']:
            mean_misere = 0
            mean_beam = 0
            mean_seqscout = 0
            mean_mcts = 0

            for j in range(xp_repeat):
                results_misere = pool.apply_async(misere, (data, target),
                                                  {'time_budget': TIME_BUDGET_XP, 'quality_measure': measure,
                                                   'iterations_limit': nb_iterations})
                results_beam = pool.apply_async(beam_search,
                                                (data, target),
                                                {'enable_i': enable_i,
                                                 'time_budget': TIME_BUDGET_XP, 'quality_measure': measure,
                                                 'iterations_limit': nb_iterations})

                result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                                   {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                    'quality_measure': measure, 'iterations_limit': nb_iterations})

                results_mcts = pool.apply_async(launch_mcts, (data, target),
                                                {'time_budget': TIME_BUDGET_XP, 'iterations_limit': nb_iterations,
                                                 'quality_measure': measure})

                results_misere = results_misere.get()
                results_beam = results_beam.get()
                result_ucb_opti = result_ucb_opti.get()
                results_mcts = results_mcts.get()

                if len(results_misere) < TOP_K:
                    print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                       len(results_misere)))
                if len(results_beam) < TOP_K:
                    print(
                        "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                          len(results_beam)))
                if len(result_ucb_opti) < TOP_K:
                    print(
                        "Too few example on ucb on dataset {}: {} results".format(datasets_names[i],
                                                                                  len(result_ucb_opti)))
                if len(results_mcts) < TOP_K:
                    print(
                        "Too few example on mctsextend on dataset {}: {} results".format(datasets_names[i],
                                                                                         len(results_mcts)))

                mean_misere += average_results(results_misere)
                mean_beam += average_results(results_beam)
                mean_seqscout += average_results(result_ucb_opti)
                mean_mcts += average_results(results_mcts)

            mean_misere = mean_misere / xp_repeat
            mean_beam = mean_beam / xp_repeat
            mean_seqscout = mean_seqscout / xp_repeat
            mean_mcts = mean_mcts / xp_repeat

            print('For datasets {}, measure {}, algorithm misere the means score is: {}'.format(datasets_names[i],
                                                                                                measure, mean_misere))
            print('For datasets {}, measure {}, algorithm beam_search the means score is: {}'.format(datasets_names[i],
                                                                                                     measure,
                                                                                                     mean_beam))
            print('For datasets {}, measure {}, algorithm ucb the means score is: {}'.format(datasets_names[i],
                                                                                             measure,
                                                                                             mean_seqscout))
            print('For datasets {}, measure {}, algorithm mctsextend the means score is: {}'.format(datasets_names[i],
                                                                                                    measure,
                                                                                                    mean_mcts))


def quality_over_theta():
    number_dataset = 1
    data, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=5, maxtasksperchild=1)

    # if we want to average
    nb_launched = 5

    theta = 0.1

    data_final = {'WRAcc': [], 'theta': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP, 'theta': theta})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP, 'theta': theta})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'theta': theta})

            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP, 'theta': theta})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_beam) < TOP_K:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print("Too few ucb: {}".format(len(result_ucb_opti)))
            if len(results_misere) < TOP_K:
                print("Too few misere: {}".format(len(results_misere)))
            if len(results_mcts) < TOP_K:
                print("Too few mctsextent : {}".format(len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), theta=theta, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), theta=theta, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), theta=theta,
                             Algorithm='SeqScout')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts)), theta=theta,
                             Algorithm='MCTSExtent')

        theta += 0.1

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='theta', y='WRAcc', hue='Algorithm')
    plt.savefig('./theta/over_theta.png')

    df.to_pickle('./theta/result')

    if SHOW:
        plt.show()


def quality_over_top_k():
    number_dataset = 3
    data, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=3, maxtasksperchild=1)

    # if we want to average
    nb_launched = 5
    top_k = 1

    data_final = {'WRAcc': [], 'top_k': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'top_k': top_k, 'time_budget': TIME_BUDGET_XP})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'top_k': top_k,
                                             'time_budget': TIME_BUDGET_XP})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'top_k': top_k,
                                                'time_budget': TIME_BUDGET_XP})

            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP, 'top_k': top_k})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_beam) < top_k:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < top_k:
                print("Too few SeqScout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < top_k:
                print("Too few misere: {}".format(len(results_misere)))
            if len(results_mcts) < top_k:
                print("Too few MCTSExtent: {}".format(len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), top_k=top_k, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), top_k=top_k, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), top_k=top_k,
                             Algorithm='SeqScout')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts)), top_k=top_k,
                             Algorithm='MCTSExtent')

        top_k += 10

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='top_k', y='WRAcc', hue='Algorithm')
    # ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./top_k/over_top_k.png')
    df.to_pickle('./top_k/result')

    if SHOW:
        plt.show()


def quality_over_size():
    number_dataset = 6
    data_origin, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=3, maxtasksperchild=1)

    # if we want to average
    nb_launched = 5

    size = 15
    size_step = 4
    data_final = {'WRAcc': [], 'size': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        data = reduce_k_length(size, data_origin)
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP})

            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': TIME_BUDGET_XP})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

            results_mcts = pool.apply_async(launch_mcts, (data, target),
                                            {'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_beam) < TOP_K:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print("Too few SeqScout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < TOP_K:
                print("Too few misere: {}".format(len(results_misere)))
            if len(results_mcts) < TOP_K:
                print("Too few MCTSExtent: {}".format(len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), size=size, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), size=size, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), size=size,
                             Algorithm='SeqScout')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_mcts)), size=size,
                             Algorithm='MCTSExtent')

        size += size_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='size', y='WRAcc', hue='Algorithm')
    ax.set(xlabel='Length max', ylabel='WRAcc')

    # ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./space_size/over_size.png')
    df.to_pickle('./space_size/result')

    if SHOW:
        plt.show()


def number_iterations_optima():
    iterations_limit = 100

    # if we want to average
    nb_launched = 5

    data_final = {'cost': [], 'iterations': [], 'dataset_name': []}
    for j, (data, target, enable_i) in enumerate(datasets):
        for i in range(nb_launched):
            # we reset the count of iterations
            seqscout.global_var.ITERATION_NUMBER = 0

            result_ucb_opti = seq_scout(data, target, enable_i=enable_i, time_budget=TIME_BUDGET_XP,
                                        iterations_limit=iterations_limit, vertical=False)

            if len(result_ucb_opti) < TOP_K:
                print("Too few SeqScout: {}".format(len(result_ucb_opti)))

            iterations = 1000

            additional_iterations = seqscout.global_var.ITERATION_NUMBER - iterations_limit

            for i in range(10):
                data_add_generic(data_final, cost=additional_iterations / iterations, iterations=iterations,
                                 dataset_name=datasets_names[j])
                iterations += 2000

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='cost', hue='dataset_name')

    plt.savefig('./number_iterations_optima/it_optima.png')
    df.to_pickle('./number_iterations_optima/result')

    if SHOW:
        plt.show()


def add_lengths(patterns, dataset_name, data_final, algo):
    for pattern in patterns:
        k_length_p = k_length(pattern[1])
        data_add_generic(data_final, Length=k_length_p, dataset=dataset_name, Algorithm=algo)


def boxplots_description_lengths():
    pool = Pool(processes=3, maxtasksperchild=1)

    data_final = {'Length': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        results_misere = pool.apply_async(misere, (data, target),
                                          {'time_budget': TIME_BUDGET_XP})
        results_beam = pool.apply_async(beam_search,
                                        (data, target),
                                        {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

        result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                           {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP, 'vertical': False})

        results_mcts = pool.apply_async(launch_mcts, (data, target),
                                        {'time_budget': TIME_BUDGET_XP})

        results_misere = results_misere.get()
        results_beam = results_beam.get()
        result_ucb_opti = result_ucb_opti.get()
        results_mcts = results_mcts.get()

        if len(results_misere) < TOP_K:
            print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                               len(results_misere)))
        if len(results_beam) < TOP_K:
            print(
                "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                  len(results_beam)))
        if len(result_ucb_opti) < TOP_K:
            print(
                "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                               len(result_ucb_opti)))
        if len(results_mcts) < TOP_K:
            print(
                "Too few example on MCTSExtent on dataset {}: {} results".format(datasets_names[i],
                                                                                 len(results_mcts)))

        add_lengths(results_misere, datasets_names[i], data_final, 'misere')
        add_lengths(results_beam, datasets_names[i], data_final, 'beam')
        add_lengths(result_ucb_opti, datasets_names[i], data_final, 'SeqScout')
        add_lengths(results_mcts, datasets_names[i], data_final, 'MCTSExtent')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.boxplot(x='dataset', y='Length', hue='Algorithm', data=df)
    plt.savefig('./lengths/boxplot.png')
    df.to_pickle('./lengths/result')

    if SHOW:
        plt.show()

def optimizer(data, target, alg_name, time_budget, iteration_limit, enable_i):
    if alg_name == 'misere':
        results = misere(data, target, time_budget=time_budget, iterations_limit=iteration_limit)
    elif alg_name == 'BeamSearch':
        results = beam_search(data, target, time_budget=time_budget, iterations_limit=iteration_limit, enable_i=enable_i)
    elif alg_name == 'SeqScout':
        results = seq_scout(data, target, time_budget=time_budget, iterations_limit=iteration_limit, enable_i=enable_i)
    elif alg_name == 'MCTSExtent':
        results = launch_mcts(data, target, time_budget=time_budget, iterations_limit=iteration_limit)
    else:
        print('Error algo name')
        return

    sorted_patterns = PrioritySet(theta=THETA)
    for result in results:
        sorted_patterns.add(result[1], result[0])

    results_post_opti = optimize_pattern(results, extract_items(data), data, [], target, TOP_K, sorted_patterns, enable_i)

    return results, results_post_opti


def barplot_increase_local_optima(it_number):
    pool = Pool(processes=5, maxtasksperchild=1)
    xp_repeat = 5

    data_final = {'WRAcc': [], 'WRAcc_opti': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        for j in range(xp_repeat):
            results_misere = pool.apply_async(optimizer, (data, target, 'misere', TIME_BUDGET_XP, it_number, enable_i))
            results_beam = pool.apply_async(optimizer, (data, target, 'BeamSearch', TIME_BUDGET_XP, it_number, enable_i))
            results_ucb_opti = pool.apply_async(optimizer, (data, target, 'SeqScout', TIME_BUDGET_XP, it_number, enable_i))
            results_mcts = pool.apply_async(optimizer, (data, target, 'MCTSExtent', TIME_BUDGET_XP, it_number, enable_i))

            results_misere, results_misere_opti = results_misere.get()
            results_beam, results_beam_opti = results_beam.get()
            result_ucb_opti, results_ucb_opti = results_ucb_opti.get()
            results_mcts, results_mcts_opti = results_mcts.get()

            if len(results_misere) < TOP_K:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < TOP_K:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print(
                    "Too few example on seqscout on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))
            if len(results_mcts) < TOP_K:
                print(
                    "Too few example on mctsextend on dataset {}: {} results".format(datasets_names[i],
                                                                                     len(results_mcts)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='misere', WRAcc_opti=max(0, average_results(results_misere_opti)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='beam', WRAcc_opti=max(0, average_results(results_beam_opti)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='SeqScout', WRAcc_opti=max(0, average_results(results_ucb_opti)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='MCTSExtent', WRAcc_opti=max(0, average_results(results_mcts_opti)))


    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    df.to_pickle('./local_opti_increase/result_{}'.format(it_number))

    plt.clf()
    sns.set_color_codes("pastel")
    ax = sns.barplot(x='dataset', y='WRAcc_opti', hue='Algorithm', data=df)

    sns.set_color_codes("muted")
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)

    plt.savefig('./local_opti_increase/barplot_{}.png'.format(it_number))

    if SHOW:
        plt.show()


if __name__ == '__main__':
    # barplot_dataset_iterations()
    # barplot_dataset_improvement_iterations()
    # barplot_dataset_time()
    # quality_over_theta()
    #show_quality_over_iterations_ucb(1)
    show_quality_over_iterations_ucb(5)
    show_quality_over_iterations_ucb(6)
    show_quality_over_iterations_ucb(0)
    # show_quality_over_iterations_ucb(2)
    # show_quality_over_iterations_ucb(3)
    # show_quality_over_iterations_ucb(4)
    # compare_ground_truth()
    # quality_over_top_k()
    # quality_over_size()
    # naive_vs_bitset_seqscout()
    # number_iterations_optima()
    # boxplots_description_lengths()
    # other_measures()
    # barplot_increase_local_optima(1000)
    #barplot_increase_local_optima(3000)
    #barplot_increase_local_optima(6000)
    #barplot_increase_local_optima(10000)

