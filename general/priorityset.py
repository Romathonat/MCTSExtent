import heapq
import general.conf as conf

from general.utils import is_subsequence, sequence_mutable_to_immutable


def jaccard_measure_misere(sequence1, sequence2, data):
    intersection = 0
    union = 0
    for sequence in data:
        sequence = sequence[1:]
        sequence = sequence_mutable_to_immutable(sequence)
        seq1 = False
        seq2 = False

        if is_subsequence(sequence1, sequence):
            seq1 = True
        if is_subsequence(sequence2, sequence):
            seq2 = True

        if seq1 or seq2:
            union += 1

        if seq1 and seq2:
            intersection += 1

    try:
        return intersection / union
    except ZeroDivisionError:
        return 0


def filter_results(results, data, theta, k):
    """
    Filter redundant elements
    :param results: must be a node
    :param theta:
    :return: filtered list
    """

    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []

    for i, result in enumerate(results_list):
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure_misere(result[1],
                                      filtered_element[1], data) > theta:
                similar = True

        if not similar:
            filtered_elements.append(result)

        if len(filtered_elements) > k:
            break

    return filtered_elements


def filter_results_not_singleton(results, data, theta, k):
    """
    Filter redundant elements that are not singletons
    :param results: must be a node
    :param theta:
    :return: filtered list
    """

    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []

    for i, result in enumerate(results_list):
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure_misere(result[1],
                                      filtered_element[1], data) > theta:
                similar = True

        if not similar and len(result[1]) > 1:
            filtered_elements.append(result)

        if len(filtered_elements) > k:
            break

    return filtered_elements


class PrioritySet(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """

    def __init__(self, k=conf.TOP_K, theta=conf.THETA):
        self.k = k
        self.heap = []
        self.set = set()
        self.theta = theta

    def add(self, sequence, wracc):
        if sequence not in self.set:
            heapq.heappush(self.heap, (wracc, sequence))
            self.set.add(sequence)

    def add_preserve_memory(self, sequence, wracc, data):
        self.add(sequence, wracc)

        # we remove elements that are not in top_k
        self.heap = self.get_top_k(self.k)

        ### UGLY ###
        set_top_k = set()

        for _, seq in self.heap:
            set_top_k.add(seq)

        self.set = set_top_k

    def get(self):
        wracc, sequence = heapq.heappop(self.heap)
        self.set.remove(sequence)
        return (wracc, sequence)

    def get_top_k(self, k):
        data = heapq.nlargest(k, self.heap)
        return data

    def get_top_k_non_redundant(self, data, k):
        filtered_result = filter_results(self.heap, data, self.theta, k)
        return heapq.nlargest(k, filtered_result)

    def get_top_k_non_redundant_non_singleton(self, data, k):
        filtered_result = filter_results_not_singleton(self.heap, data, self.theta, k)
        return heapq.nlargest(k, filtered_result)


class PrioritySetUCB(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """

    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, sequence, tuple):
        '''
        :param sequence:
        :param tuple: (UCB, Ni, WRAcc)
        :return:
        '''
        if sequence not in self.set:
            # we use - sign because heapq return the smalest element
            heapq.heappush(self.heap, (-tuple[0], tuple[1], tuple[2], sequence))
            self.set.add(sequence)

    def pop(self):
        '''
        :return: the max element
        '''
        UCB, Ni, wracc, sequence = heapq.heappop(self.heap)
        self.set.remove(sequence)
        return (-UCB, Ni, wracc, sequence)
