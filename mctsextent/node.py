import random
import general.conf as conf
import sys

from general.utils import find_LCS, sequence_mutable_to_immutable, compute_quality_extend, k_length, is_subsequence


# sys.setrecursionlimit(15000)

class Node():
    def __init__(self, intent, parent, data, data_positive, target_class, node_hashmap, quality_measure=conf.QUALITY_MEASURE):
        '''
        :param added_object:
        :param extend: identifiers of objects
        :param parent:
        :param data:
        :param target_class:
        '''

        self.intent = intent
        self.data = data
        self.data_positive = data_positive
        self.node_hashmap = node_hashmap

        # the extend is the id of sequences
        self.quality, self.extend_positive = self.get_extend_and_quality(data, self.intent, target_class, quality_measure=quality_measure)

        if parent != None:
            self.parents = [parent]
            parent.children.append(self)
        else:
            self.parents = []

        self.children = []
        self.candidate_sequences_expand = self.compute_sequence_expand(data_positive) # dataset sequences to expand

        self.number_visits = 1
        self.dead_end = False

    def get_normalized_quality(self, quality_measure=conf.QUALITY_MEASURE):
        if quality_measure == 'WRAcc':
            return (self.quality + 0.25) * 2
        else:
            return self.quality

    def get_extend_and_quality(self, data, subsequence, target_class, quality_measure=conf.QUALITY_MEASURE):
        if self.intent == None:
            return 0, []
        return compute_quality_extend(data, subsequence, target_class, quality_measure=quality_measure)

    def compute_sequence_expand(self, data_positive):
        # we cannot add sequences wich are supersequences of pattern, or else the LCS will return the same node, creating a dag and many problems !
        try:
            return [seq[1:] for i, seq in enumerate(data_positive) if
                    i not in self.extend_positive and not is_subsequence(self.intent, seq[1:])]
        except TypeError:
            return [seq[1:] for i, seq in enumerate(data_positive) if i not in self.extend_positive]

    def is_fully_expanded(self):
        return len(self.candidate_sequences_expand) == 0

    def is_terminal(self):
        # a node is terminal if all positive sequences have been explored
        return len(self.extend_positive) == len(self.data_positive)

    def is_dead_end(self):
        '''
        A terminal node is a dead end, and a node with all its children being dead ends is a dead end
        '''
        if self.is_terminal() or self.dead_end:
            self.dead_end = True
            return True

        if not self.is_fully_expanded():
            # a node non-fully expanded is not a dead end
            return False

        for child in self.children:
            # improving time computing
            if not child.is_dead_end():
                return False

        self.dead_end = True
        return True

    def expand(self, data, data_positive, target_class, quality_measure=conf.QUALITY_MEASURE):
        random_object = random.sample(self.candidate_sequences_expand, 1)[0]
        self.candidate_sequences_expand.remove(random_object)

        if self.intent == None:
            # for the root node, it is directly database sequences
            sequence_children = sequence_mutable_to_immutable(random_object)
        else:
            sequence_children = sequence_mutable_to_immutable(find_LCS(random_object, self.intent))

        if sequence_children in self.node_hashmap:
            child = self.node_hashmap[sequence_children]
            child.parents.append(self)
            self.children.append(child)
        else:
            child = Node(sequence_children, self, data, data_positive, target_class, self.node_hashmap, quality_measure=quality_measure)
            self.node_hashmap[sequence_children] = child

        return child

    def update(self, reward):
        """
        Update the quality of the node
        :param reward: the roll-out score
        :return: None
        """
        # Mean-update
        self.quality = (self.number_visits * self.quality + reward) / (
                self.number_visits + 1)
        self.number_visits += 1
