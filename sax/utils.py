import random
from general.utils import is_subsequence, subsequence_indices

def draw_pattern_instance(data, pattern, kmeans, plt):
    dat = {'y': [], 'x': [], 'pattern': []}

    # we need to add the class to make it work with compute_quality_extend
    instances = []
    for sequence in data:
        if is_subsequence(pattern, sequence):
            instances.append(sequence)

    instances = random.sample(instances, 5)

    for instance in instances:
        instance = instance[1:]
        indices = subsequence_indices(pattern, instance)

        x, y = [x for x, y in enumerate(instance)], [kmeans.cluster_centers_[list(y)[0]] for y in instance]
        plt.figure()
        plt.scatter(x, y)
        for i, y_value in enumerate(y):
            if i in indices:
                plt.scatter(i, y_value, c='red')

    plt.show()

def draw_ts(instance, kmeans, title, plt):
    x, y = [x for x, y in enumerate(instance)], [kmeans.cluster_centers_[list(y)[0]] for y in instance]
    fig = plt.figure()
    plt.scatter(x, y)
    fig.suptitle(title, fontsize=20)


def launch_mcts_ts(DATA, target_class, kmeans, plt):
    nb_fig = 3
    count_pos = 0
    count_neg = 0

    for line in DATA:
        if line[0] == target_class and count_pos < nb_fig:
            draw_ts(line[1:], kmeans, target_class, plt)
            count_pos += 1
        elif count_neg < nb_fig:
            draw_ts(line[1:], kmeans, '-', plt)
            count_neg += 1

    '''
    results = launch_mcts(DATA, target_class, time_budget=12000, top_k=5, iterations_limit=1000)

    print_results(results)
    print_pattern_discretization(results, kmeans)

    draw_pattern_instance(DATA, results[1][1], kmeans, plt)
    '''
