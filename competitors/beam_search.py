import datetime
import functools

from general.priorityset import PrioritySet
from general.reader import read_data
from general.utils import create_s_extension, create_i_extension, extract_items, print_results, \
    compute_quality

import general.conf as conf


def compare_sequences(x, y):
    if sorted(sorted(list(x[0]))) > sorted(sorted(list(y[0]))):
        return True
    elif len(x) > 1 and len(y) > 1:
        return compare_sequences(x[1:], y[1:])
    else:
        return False


def compute_children(sequence, items, enable_i=True):
    """
    :param sequence: the sequence we want to compute children of
    :param items: list of existing items
    :param enable_i: enable i_extensions or not. Useful when sequences are singletons like DNA
    :return: the set of sequences that we can generate from the current one
    NB: We convert to mutable/immutable object in order to have a set of subsequences,
    which automatically removes duplicates
    """
    new_subsequences = list()

    for item in items:
        for index, itemset in enumerate(sequence):
            s_extension = create_s_extension(sequence, item, index)

            if s_extension not in new_subsequences:
                new_subsequences.append(s_extension)

            if enable_i:
                pseudo_i_extension = create_i_extension(sequence, item,
                                                        index)

                length_i_ext = sum([len(i) for i in pseudo_i_extension])
                len_subsequence = sum([len(i) for i in sequence])

                # we prevent the case where we add an existing element to itemset
                if (length_i_ext > len_subsequence) and pseudo_i_extension not in new_subsequences:
                    new_subsequences.append(pseudo_i_extension)

        s_extension = create_s_extension(sequence, item, len(sequence))

        if s_extension not in new_subsequences:
            new_subsequences.append(s_extension)

    # we need to sort with tuples of frozensets
    return sorted(new_subsequences, key=functools.cmp_to_key(compare_sequences))


def items_to_sequences(items):
    sequences = []
    for item in items:
        sequences.append((frozenset([item]),))

    return sequences


def beam_search(data, target_class, time_budget=conf.TIME_BUDGET, enable_i=True, top_k=conf.TOP_K,
                beam_width=conf.BEAM_WIDTH,
                iterations_limit=conf.ITERATIONS_NUMBER,
                theta=conf.THETA,
                quality_measure=conf.QUALITY_MEASURE,
                diverse=True):
    items = extract_items(data)

    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    # candidate_queue = items_to_sequences(items)
    candidate_queue = [[]]

    sorted_patterns = PrioritySet(top_k, theta=theta)

    nb_iteration = 0
    while datetime.datetime.utcnow() - begin < time_budget and nb_iteration < iterations_limit:
        beam = PrioritySet()

        while (len(candidate_queue) != 0) and nb_iteration < iterations_limit:
            seed = candidate_queue.pop(0)
            children = compute_children(seed, items, enable_i)

            for child in children:
                if nb_iteration >= iterations_limit:
                    break

                quality = compute_quality(data, child, target_class, quality_measure=quality_measure)

                # sorted_patterns.add_preserve_memory(child, quality, data)
                sorted_patterns.add(child, quality)
                beam.add(child, quality)
                nb_iteration += 1
        if diverse:
            candidate_queue = [j for i, j in beam.get_top_k_non_redundant(data, beam_width)]
        else:
            candidate_queue = [j for i, j in beam.get_top_k(beam_width)]

    # print("Number iterations beam search: {}".format(nb_iteration))
    return sorted_patterns.get_top_k_non_redundant(data, top_k)


def launch():
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    # DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')
    # DATA = read_jmlr('svm', pathlib.Path(__file__).parent.parent / 'data/jmlr/jmlr')
    DATA = read_data('../data/splice.data')
    # DATA = read_data_kosarak('../data/easy.data')
    # DATA = read_data_kosarak('../data/skating.data')

    # DATA, items_dict = read_retail('71270', pathlib.Path(__file__).parent.parent / 'data/retail.csv')
    # print("Target item:{}".format(items_dict['71270']))

    # DATA = read_data_kosarak('../data/figures_rc.dat')
    # DATA = read_data_kosarak(pathlib.Path(__file__).parent.parent / 'data/context.data')

    results = beam_search(DATA, 'EI', time_budget=100000000, enable_i=False, top_k=5, iterations_limit=10000,
                          beam_width=10, quality_measure='WRAcc')
    print_results(results)


if __name__ == '__main__':
    launch()
