import datetime
import sys
import random
import copy

import math

import general.conf as conf

from general.reader import read_data_kosarak
from general.utils import sequence_mutable_to_immutable, print_results, compute_quality, \
    sequence_immutable_to_mutable, print_rocket_league, filter_positive, filter_empty_sequences, encode_items, \
    encode_data, print_results_decode, extract_items, decode_sequences

from general.priorityset import PrioritySet
from mctsextent.node import Node

sys.setrecursionlimit(15000)


def best_child(node):
    """
    Returns the best child node of node w.r.t UCB
    :param node:
    :return:
    """
    if node.is_dead_end() and len(node.parents) == 0:
        # root is a dead end, we have finished
        return 'finished'

    best_node = None
    max_score = -float("inf")

    for child in node.children:
        current_ucb = child.get_normalized_quality(conf.QUALITY_MEASURE) / child.number_visits + 0.5 * math.sqrt(
            2 * math.log(node.number_visits) / child.number_visits)

        if current_ucb > max_score and not child.is_dead_end():
            max_score = current_ucb
            best_node = child

    if best_node == None:
        # if program reaches here, the node is a dead_end, we go to the parent
        return node.parent

    return best_node


def select(node):
    """
    Select the best node, using exploration-exploitation tradeoff
    :param node: the node from where we begin to search
    :return: the selected node, or None if exploration is finished
    """
    while node != 'finished':
        if not node.is_fully_expanded():
            return node
        else:
            node = best_child(node)

    return 'finished'


def roll_out(node, data, target_class, quality_measure=conf.QUALITY_MEASURE):
    """
    Generalize a sequence by deleting random items
    :param node: the node corresponding to the sequence to generalize
    :param data:
    :param target_class:
    :param quality_measure:
    :return: the new sequence and its quality
    """
    sequence = copy.deepcopy(node.intent)
    sequence = sequence_immutable_to_mutable(sequence)

    # we remove z items randomly
    seq_items_nb = len([i for j_set in sequence for i in j_set])

    z = random.randint(0, seq_items_nb)

    for _ in range(z):
        chosen_itemset_i = random.randint(0, len(sequence) - 1)
        chosen_itemset = sequence[chosen_itemset_i]

        chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

        if len(chosen_itemset) == 0:
            sequence.pop(chosen_itemset_i)

    reward = compute_quality(data, sequence, target_class, quality_measure=quality_measure)

    return sequence, reward


def update(node, reward):
    """
    Backtrack: update the node and recursively update all parent nodes until the extent root
    :param node: the node we want to update
    :param reward: the reward we got
    :return: None
    """
    # with python we have a limit to the recursive approach. We will do it with a BFS
    # node.update(reward)
    #
    # for parent in node.parents:
    #     if parent != None:
    #         update(parent, reward)

    update_nodes = {node}
    parents_seen = set()

    while len(update_nodes) != 0:
        node = random.sample(update_nodes, 1)[0]
        parents_seen.add(node)
        for parent in node.parents:
            if parent not in parents_seen:
                update_nodes.add(parent)

        node.update(reward)
        update_nodes.remove(node)


def get_patterns(path='', target_class='+', top_k=5, time_budget=10, theta=0.8):
    '''
    :param path: path to the file containing data, in kosarak format
    :param target_class: the target class we want to find pattern of: string
    :param top_k: the number of patterns we want to get
    :param time_budget: the time we give to the algorithm
    :return: the top-k best pattern w.r.t WRAcc, and display them
    '''
    data = read_data_kosarak(path)
    items = extract_items(data)
    items, items_to_encoding, encoding_to_items = encode_items(items)
    data = encode_data(data, items_to_encoding)

    results = launch_mcts(data, target_class, top_k=top_k, time_budget=time_budget, theta=theta,
                          iterations_limit=2 ** 30)
    print_results_decode(results, encoding_to_items)

    return decode_sequences(results, encoding_to_items)


def launch_mcts(data, target_class, time_budget=conf.TIME_BUDGET, top_k=conf.TOP_K, theta=conf.THETA,
                iterations_limit=conf.ITERATIONS_NUMBER, quality_measure=conf.QUALITY_MEASURE):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    data_positive = filter_positive(data, target_class)
    data = filter_empty_sequences(data)

    node_hashmap = {}
    root_node = Node(None, None, data, data_positive, target_class, node_hashmap)
    node_hashmap[('.')] = root_node

    sorted_patterns = PrioritySet(k=top_k, theta=theta)
    iteration_count = 0

    while datetime.datetime.utcnow() - begin <= time_budget and iteration_count < iterations_limit:
        node_sel = select(root_node)

        if node_sel == 'finished':
            print('Finished')
            break

        node_expand = node_sel.expand(data, data_positive, target_class, quality_measure=quality_measure)
        sorted_patterns.add(sequence_mutable_to_immutable(node_expand.intent), node_expand.quality)

        sequence_reward, reward = roll_out(node_expand, data, target_class, quality_measure=quality_measure)

        sorted_patterns.add(sequence_mutable_to_immutable(sequence_reward), reward)

        update(node_expand, reward)
        iteration_count += 1

        # if iteration_count % int(iterations_limit * 0.1) == 0:
        #    print('{}%'.format(iteration_count / iterations_limit * 100))

    print('Number iteration mcts: {}'.format(iteration_count))
    return sorted_patterns.get_top_k_non_redundant(data, top_k)


if __name__ == '__main__':
    results = get_patterns(path='../data/figures_rc.dat', target_class='1', top_k=10, theta=0.5)
    ''' 
    DATA = read_data_kosarak('../data/figures_rc.dat')

    # DATA, kmeans = readECG()
    # DATA, kmeans = read_gun_point()

    target_class = '4'

    # launch_mcts_ts(DATA, target_class, kmeans)
    results = launch_mcts(DATA, target_class, top_k=3, iterations_limit=20000)

    # print_results(results)
    print_rocket_league(results)
    '''


