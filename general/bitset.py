import math

import seqscout
from general import conf
from general.utils import k_length


def compute_first_zero_mask(data_length, bitset_slot_size):
    first_zero = 2 ** (bitset_slot_size - 1) - 1
    first_zero_mask = 0

    for i in range(data_length):
        first_zero_mask |= first_zero << i * bitset_slot_size

    return first_zero_mask


def compute_last_ones_mask(data_length, bitset_slot_size):
    last_ones = 1
    last_ones_mask = 1

    for i in range(data_length):
        last_ones_mask |= last_ones << i * bitset_slot_size

    return last_ones_mask


def following_ones(bitset, bitset_slot_size, first_zero_mask):
    """
    Transform bitset with 1s following for each 1 encoutered, for
    each bitset_slot.
    :param bitset:
    :param bitset_slot_size: the size of a slot in the bitset
    :return: a bitset (number)
    """
    # the first one needs to be a zero
    bitset = bitset >> 1
    bitset = bitset & first_zero_mask

    temp = bitset >> 1
    temp = temp & first_zero_mask

    bitset |= temp

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    return bitset


def get_support_from_vector(bitset, bitset_slot_size, first_zero_mask,
                            last_ones_mask):
    temp = bitset >> 1
    temp = temp & first_zero_mask

    bitset |= temp

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    bitset = bitset & last_ones_mask

    i = bitset.bit_length()

    data_length = math.ceil(i / bitset_slot_size)

    bitset_simple = 0
    count = 0

    while i > 0:
        if bitset >> (i - 1) & 1:
            bitset_simple |= 1 << (data_length - count - 1)

        count += 1
        i -= bitset_slot_size

    # now we have a vector with ones or 0 at the end of each slot. We just need to
    # compute the hamming distance
    return hamming_weight(bitset_simple), bitset_simple


def generate_bitset(itemset, data, bitset_slot_size):
    """
    Generate the bitset of itemset

    :param itemset: the itemset we want to get the bitset
    :param data: the dataset
    :return: the bitset of itemset
    """
    bitset = 0

    # we compute the extend by scanning the database
    for line in data:
        line = line[1:]
        sequence_bitset = 0
        for itemset_line in line:
            if itemset.issubset(itemset_line):
                bit = 1
            else:
                bit = 0

            sequence_bitset |= bit
            sequence_bitset = sequence_bitset << 1

        # for last element we need to reshift
        sequence_bitset = sequence_bitset >> 1

        # we shift to complete with 0
        sequence_bitset = sequence_bitset << bitset_slot_size - (len(line))

        # we add this bit vector to bitset
        bitset |= sequence_bitset
        bitset = bitset << bitset_slot_size

    # for the last element we need to reshift
    bitset = bitset >> bitset_slot_size

    return bitset


def compute_bitset_slot_size(data):
    max_size_itemset = 1

    for line in data:
        max_size_line = len(max(line, key=lambda x: len(x)))
        if max_size_line > max_size_itemset:
            max_size_itemset = max_size_line

    return max_size_itemset

def hamming_weight(vector):
    w = 0
    while vector:
        w += 1
        vector &= vector - 1
    return w


def compute_quality_vertical(data, subsequence, target_class, bitset_slot_size,
                             itemsets_bitsets, class_data_count, first_zero_mask,
                             last_ones_mask, quality_measure=conf.QUALITY_MEASURE):
    seqscout.global_var.increase_it_number()
    length = k_length(subsequence)
    bitset = 0

    if length == 0:
        # the empty node is present everywhere
        # we just have to create a vector of ones
        bitset = 2 ** (len(data) * bitset_slot_size) - 1
    elif length == 1:
        singleton = frozenset(subsequence[0])
        bitset = generate_bitset(singleton, data,
                                 bitset_slot_size)
        itemsets_bitsets[singleton] = bitset
    else:
        # general case
        bitset = 2 ** (len(data) * bitset_slot_size) - 1
        first_iteration = True
        for itemset_i in subsequence:
            itemset = frozenset(itemset_i)

            try:
                itemset_bitset = itemsets_bitsets[itemset]
            except KeyError:
                # the bitset is not in the hashmap, we need to generate it
                itemset_bitset = generate_bitset(itemset, data,
                                                 bitset_slot_size)
                itemsets_bitsets[itemset] = itemset_bitset

            if first_iteration:
                first_iteration = False

                # aie aie aie !
                bitset = itemset_bitset
            else:
                bitset = following_ones(bitset, bitset_slot_size,
                                        first_zero_mask)

                bitset &= itemset_bitset

    # now we just need to extract support, supersequence and class_pattern_count
    class_pattern_count = 0

    support, bitset_simple = get_support_from_vector(bitset,
                                                     bitset_slot_size,
                                                     first_zero_mask,
                                                     last_ones_mask)

    # find supersequences and count class pattern:
    i = bitset_simple.bit_length() - 1

    while i >= 0:
        if bitset_simple >> i & 1:
            index_data = len(data) - i - 1

            if data[index_data][0] == target_class:
                class_pattern_count += 1

        i -= 1

    occurency_ratio = support / len(data)

    if quality_measure == 'WRAcc':
        # we find the number of elements who have the right target_class
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return -0.25, 0

        class_data_ratio = class_data_count / len(data)
        wracc = occurency_ratio * (class_pattern_ratio - class_data_ratio)
        return wracc, bitset

    elif quality_measure == 'Informedness':
        tn = len(data) - support - (class_data_count - class_pattern_count)

        tpr = class_pattern_count / (class_pattern_count + (class_data_count - class_pattern_count))

        tnr = tn / (class_pattern_count + tn)
        return tnr + tpr - 1, bitset

    elif quality_measure == 'F1':
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return 0, 0
        precision = class_pattern_ratio
        recall = class_pattern_count / class_data_count
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0
        return f1, bitset
    else:
        raise ValueError('The quality measure name is not valid')
