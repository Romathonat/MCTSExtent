import copy

import general.conf as conf


def increase_it_number():
    global ITERATION_NUMBER
    ITERATION_NUMBER += 1


def sequence_mutable_to_immutable(sequence):
    """
    :param sequence: form [{}, {}, ...]
    :return: the same sequence in its immutable form
    """
    return tuple([frozenset(i) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    """
    :param sequence: form (frozenset(), frozenset(), ...)
    :return: the same sequence in its mutable form
    """
    return [set(i) for i in sequence]


def create_s_extension(sequence, item, index):
    """
    Perform an s-extension
    :param sequence: the sequence we are extending
    :param item: the item to insert (not a set, an item !)
    :param index: the index to add the item
    :return: an immutable sequence
    """
    # insert would require a deep copy, which is not performance

    new_sequence = []
    appended = False

    for i, itemset in enumerate(sequence):
        if i == index:
            new_sequence.append(frozenset({item}))
            appended = True
        new_sequence.append(itemset)

    if not appended:
        new_sequence.append(frozenset({item}))

    return tuple(new_sequence)


def create_i_extension(sequence, item, index):
    """
    Perform an i-extension
    :param sequence: the sequence we are extending
    :param item: the item to merge to(not a set, an item !)
    :param index: the index to add the item
    :return: an immutable sequence
    """
    new_sequence = []

    for i, itemset in enumerate(sequence):
        if i == index:
            new_sequence.append(frozenset({item}).union(itemset))
        else:
            new_sequence.append(itemset)

    return tuple(new_sequence)


def count_target_class_data(data, target_class):
    """
    Count the number of occurences of target_class in the data
    :param data: sequential data of for [[class, {}, {} ...], [class, {}, {}], ...]
    :param target_class: the targeted class
    :return: the count
    """
    count = 0
    for row in data:
        if row[0] == target_class:
            count += 1

    return count


def k_length(sequence):
    """
    :param sequence: the considered sequence
    :return: the length of the sequence
    """
    return sum([len(i) for i in sequence])


def reduce_k_length(target_length, data):
    '''
    Reduce k_length of dataset removing class
    :param target_length:
    :param data:
    :return:
    '''
    new_data = []
    for i, sequence in enumerate(data):
        if k_length(sequence) > target_length:
            new_sequence = [sequence[0]]

            count = 0

            # we do not take into account the target

            for itemset in sequence[1:]:
                itemset_copy = copy.deepcopy(itemset)

                if len(itemset_copy) + count > target_length:
                    # we need to remove some elements of this itemset
                    for i in range(len(itemset_copy) + count - target_length):
                        itemset_copy.remove(max(itemset_copy))

                    if len(itemset_copy) > 0:
                        new_sequence.append(itemset_copy)
                    break
                else:
                    count += len(itemset_copy)
                    new_sequence.append(itemset_copy)
        else:
            new_sequence = sequence[:]

        new_data.append(new_sequence)
    return new_data


def is_subsequence(a, b):
    """ check if sequence a is a subsequence of b
    """
    i_a, i_b = 0, 0

    while i_a < len(a) and i_b < len(b):
        if a[i_a].issubset(b[i_b]):
            i_a += 1
        i_b += 1

    return i_a == len(a)


def subsequence_indices(a, b):
    """ Return itemset indices of b that itemset of a are included in
        Precondition: a is a subset of b
    """
    index_b_mem = 0
    indices_b = []
    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                indices_b.append(index_b)
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            return indices_b

    return indices_b


def encode_data(data, item_to_encoding):
    """
    Replaces all item in data by its encoding
    :param data:
    :param item_to_encoding:
    :return:
    """
    for line in data:
        for i, itemset in enumerate(line[1:]):
            encoded_itemset = set()
            for item in itemset:
                encoded_itemset.add(item_to_encoding[item])
            line[i + 1] = encoded_itemset

    return data


def decode_sequence(sequence, encoding_to_item):
    """
    Give the true values of sequence
    :param sequence: the sequence to decode in the form [{}, ..., {}]
    :return: the decoded sequence
    """
    return_sequence = []

    for i, itemset in enumerate(sequence):
        decoded_itemset = set()
        for item in itemset:
            decoded_itemset.add(encoding_to_item[item])
        return_sequence.append(decoded_itemset)
    return return_sequence


def decode_sequences(results, encoding_to_item):
    return_results = []
    for result in results:
        return_results.append((result[0], decode_sequence(result[1], encoding_to_item)))
    return return_results


def encode_items(items):
    item_to_encoding = {}
    encoding_to_item = {}
    new_items = set()

    for i, item in enumerate(items):
        item_to_encoding[item] = i
        encoding_to_item[i] = item
        new_items.add(i)

    return new_items, item_to_encoding, encoding_to_item


def extract_items(data):
    """
    :param data: date must be on the form [[class, {}, {}, ...], [class, {}, {}, ...]]
    :return: set of items extracted
    """
    items = set()
    for sequence in data:
        for itemset in sequence[1:]:
            for item in itemset:
                items.add(item)
    return sorted(list(items))


def print_results(results):
    sum_result = 0
    for result in results:
        pattern_display = ''
        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        sum_result += result[0]

        print('Quality: {}, Pattern: {}'.format(result[0], pattern_display))

    print('Average score :{}'.format(sum_result / len(results)))


def print_results_retails(results, items_dict):
    sum_result = 0
    for result in results:
        pattern_display = ''
        for itemset in result[1]:
            itemset_display = '{'
            for item in itemset:
                itemset_display += items_dict[item] + ', '
            itemset_display += '} '
            pattern_display += itemset_display

        sum_result += result[0]

        print('Quality: {}, Pattern: {}'.format(result[0], pattern_display))

    print('Average score :{}'.format(sum_result / len(results)))


def print_results_mcts(results, encoding_to_items):
    sum_result = 0
    for result in results:
        pattern_display = ''

        sequence = decode_sequence(result[1].sequence, encoding_to_items)
        for itemset in sequence:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))
        sum_result += result[0]

    print('Average score :{}'.format(sum_result / len(results)))


def print_results_decode(results, encoding_to_items):
    decoded_results = []
    for result in results:
        decoded_result = []
        decoded_result.append(result[0])
        decoded_result.append(decode_sequence(result[1], encoding_to_items))
        decoded_results.append(decoded_result)

    print_results(decoded_results)


def average_results(results):
    sum_result = 0
    for result in results:
        sum_result += result[0]

    return sum_result / len(results)


def extract_l_max(data):
    lmax = 0
    for line in data:
        lmax = max(lmax, k_length(line))
    return lmax


def get_quality(quality_measure, class_pattern_count, support, data, class_data_count):
    if quality_measure == 'WRAcc':
        # we find the number of elements who have the right target_class
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return -0.25

        class_data_ratio = class_data_count / len(data)
        wracc = support / len(data) * (class_pattern_ratio - class_data_ratio)
        return wracc

    elif quality_measure == 'Informedness':
        tn = len(data) - support - (class_data_count - class_pattern_count)
        tpr = class_pattern_count / (class_pattern_count + (class_data_count - class_pattern_count))
        tnr = tn / (support - class_pattern_count + tn)
        return tnr + tpr - 1

    elif quality_measure == 'F1':
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return 0
        precision = class_pattern_ratio
        recall = class_pattern_count / class_data_count
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        return f1
    elif quality_measure == 'Precision':
        try:
            if support > conf.PRECISION_MIN_SUPPORT:
                precision = class_pattern_count / support
            else:
                return 0
        except ZeroDivisionError:
            return 0
        return precision
    elif quality_measure == 'Lift':
        try:
            if support > conf.PRECISION_MIN_SUPPORT:
                return class_pattern_count * len(data) / (class_data_count * support)
            else:
                return 0
        except ZeroDivisionError:
            return 0
    else:
        raise ValueError('The quality measure name is not valid')


def compute_support(data, subsequence):
    support = 0

    for sequence in data:
        sequence = sequence[1:]

        if is_subsequence(subsequence, sequence):
            support += 1

    return support


def print_rocket_league(patterns):
    '''
    :param patterns: results of mctsextend
    :return:
    '''
    translator = {'f': 'left', 'g': 'right', 'r': 'jump', 't': 'boost', 'y': 'slide', 'u': 'camera', 'j': 'down',
                  'h': 'up', 'i': 'rotate', 'k': 'accelerate', 'l': 'backward'}

    for pattern in patterns:
        quality, pattern = pattern[0], pattern[1]
        pattern_display = ''
        for itemset in pattern:
            translated_itemset = set()
            for item in itemset:
                translated_itemset.add(translator[item])
            pattern_display += repr(set(translated_itemset))

        print('Quality: {}, Pattern: {}'.format(quality, pattern_display))


def compute_quality(data, subsequence, target_class, quality_measure=conf.QUALITY_MEASURE):
    seqscout.global_var.increase_it_number()
    support = 0
    class_pattern_count = 0
    class_data_count = 0

    for sequence in data:
        current_class = sequence[0]
        sequence = sequence[1:]

        if is_subsequence(subsequence, sequence):
            support += 1
            if current_class == target_class:
                class_pattern_count += 1

        if current_class == target_class:
            class_data_count += 1

    return get_quality(quality_measure, class_pattern_count, support, data, class_data_count)


def compute_quality_extend(data, subsequence, target_class, quality_measure=conf.QUALITY_MEASURE):
    '''
    :return: the quality and the extend of positives elements
    '''
    seqscout.global_var.increase_it_number()
    extend = []
    support = 0
    class_pattern_count = 0
    class_data_count = 0

    for i, sequence in enumerate(data):
        current_class = sequence[0]
        sequence = sequence[1:]

        if is_subsequence(subsequence, sequence):
            support += 1
            if current_class == target_class:
                extend.append(i)
                class_pattern_count += 1

        if current_class == target_class:
            class_data_count += 1

    return get_quality(quality_measure, class_pattern_count, support, data, class_data_count), extend


import seqscout.global_var


def backtrack_LCS(C, seq1, seq2, i, j, lcs):
    if i == 0 or j == 0:
        return

    inter = seq1[i - 1].intersection(seq2[j - 1])

    if inter != set():
        # these two cases check what path the DP took
        if C[i - 1][j] == C[i][j]:
            return backtrack_LCS(C, seq1, seq2, i - 1, j, lcs)
        if C[i][j - 1] == C[i][j]:
            return backtrack_LCS(C, seq1, seq2, i, j - 1, lcs)
        else:
            lcs.insert(0, inter)
            return backtrack_LCS(C, seq1, seq2, i - 1, j - 1, lcs)

    if C[i][j - 1] > C[i - 1][j]:
        return backtrack_LCS(C, seq1, seq2, i, j - 1, lcs)
    else:
        return backtrack_LCS(C, seq1, seq2, i - 1, j, lcs)


def find_LCS(seq1, seq2, all=False):
    """
    find the longest common subsequence. We here consider sequences of itemsets
    Cost a lot if all = True
    :param seq1:
    :param seq2:
    :return: the longest common sequence

    """
    C = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            inter = seq1[i - 1].intersection(seq2[j - 1])

            C[i][j] = max([C[i - 1][j - 1] + len(inter),
                           C[i - 1][j],
                           C[i][j - 1]])

    # now we need to backtrack the structure to get the pattern
    if all:
        all_lcs = backtrack_all_LCS(C, seq1, seq2, len(seq1), len(seq2))
        return {i[1:] for i in all_lcs}

    lcs = []
    backtrack_LCS(C, seq1, seq2, len(seq1), len(seq2), lcs)
    return lcs


def backtrack_all_LCS(C, seq1, seq2, i, j):
    if i == 0 or j == 0:
        return {(frozenset(['.']),)}

    inter = seq1[i - 1].intersection(seq2[j - 1])

    if inter != set():
        lcs = set()

        partial_lcs = backtrack_all_LCS(C, seq1, seq2, i - 1, j - 1)
        for z in partial_lcs:
            lcs.add(sequence_mutable_to_immutable(z + (inter,)))

        # if the number is the same, we have another way of reaching lcs
        if C[i][j] == C[i][j - 1]:
            partial_lcs = backtrack_all_LCS(C, seq1, seq2, i, j - 1)
            for z in partial_lcs:
                lcs.add(sequence_mutable_to_immutable(z))

        if C[i][j] == C[i - 1][j]:
            partial_lcs = backtrack_all_LCS(C, seq1, seq2, i - 1, j)
            for z in partial_lcs:
                lcs.add(sequence_mutable_to_immutable(z))
        return lcs

    lcs = set()

    if C[i][j - 1] >= C[i - 1][j]:
        lcs = lcs.union(backtrack_all_LCS(C, seq1, seq2, i, j - 1))
    if C[i][j - 1] <= C[i - 1][j]:
        lcs = lcs.union(backtrack_all_LCS(C, seq1, seq2, i - 1, j))

    return lcs


def filter_positive(data, target_class):
    return [i for i in data if i[0] == target_class]


def filter_empty_sequences(data):
    return [i for i in data if len(i[1:]) > 0]
