import pandas as pd

from sax.normalize import normalize
from sax.paa import paa
from sax.lookup_table import lookup_table, lookup_table_equal_frequency

# IMPORTANT: Id are necessary to retrieve the correct std and means

def get_position_lookup_table(lookup, value, id=''):
    '''
    :param lookup:
    :param value:
    :param id: an id to concatenate to values of sax representation
    :return: returns the closest inferior breakpoint to value, or -inf if we are in the case [-inf, breakpoint1]
    '''

    if value < lookup[0]:
        return 'a-{}'.format(id)

    if value > lookup[-1]:
        return '{}-{}'.format(chr(ord('a') + len(lookup)), id)

    for i in range(len(lookup) - 1):
        if value > lookup[i] and value < lookup[i + 1]:
            return '{}-{}'.format(chr(ord('a') + i), id)


def convert_position_to_threshold(lookup, position_symbol):
    '''
    :param lookup: lookup table
    :param position_symbol: the symbol given by check_position_lookup_table
    :return: the interval corresponding to symbol
    '''
    if position_symbol == -float('inf'):
        return '[-inf, {}]'.format(lookup[0])

    elif position_symbol == len(lookup):
        return '[{}, -inf]'.format(lookup[position_symbol])

    else:
        return '[{}, {}]'.format(lookup[position_symbol], lookup[position_symbol + 1])


def readable_pattern(pattern, lookup, original_means, original_stds, id_to_column_name=None):
    '''
    :param pattern: the pattern we want to translate
    :param lookup: lookup table
    :param original_means: list of original means to denormalize, with index corresponding to id
    :param original_stds: list of original stds to denormalize, with index corresponding to id
    :return: the readable pattern
    '''

    # first we use the lookup_table to get the interval, then we use the original_mean and original_std (lists of)
    pattern_interval = []
    for itemset in pattern:
        itemset_intervals = {}
        for item in itemset:
            item_without_id, id = item.split('-')
            id = int(id)
            look_up_position = ord(item_without_id) - ord('a')

            if id_to_column_name is not None:
                    column_name = id_to_column_name[id]

            if look_up_position == len(lookup):
                # last interval
                itemset_intervals[column_name] = [lookup[-1] * original_stds[id] + original_means[id], 'inf']
            elif look_up_position == 0:
                itemset_intervals[column_name] = ['-inf', lookup[0] * original_stds[id] + original_means[id]]
            else:
                itemset_intervals[column_name] = [lookup[look_up_position] * original_stds[id] + original_means[id],
                                         lookup[look_up_position + 1] * original_stds[id] + original_means[id]]

        pattern_interval.append(itemset_intervals)

    return pattern_interval

def sax(ts, w, a, id=''):
    '''
    :param ts: the timeserie on which we want to apply sax
    :param w: the number of slot for the paa
    :param a: the number of items of the language (the more we have the more precise we are)
    :param id: an id to concatenate to values of sax representation
    :return: the sax representation, with numbers: 1 means the value is between 1 and 2 in lookup table
    '''
    # lookup = lookup_table(a)
    lookup = lookup_table_equal_frequency(a, ts)

    ts, mean, std = normalize(ts)
    ts_paa = paa(ts, w)

    sequence = []
    for _, elt in enumerate(ts_paa):
        sequence.append(get_position_lookup_table(lookup, elt, id=id))

    return pd.Series(sequence, index=ts_paa.index), mean, std, lookup


def sax_slot_size(ts, slot_size, a, id=''):
    '''
    :param ts: the time serie
    :param slot_size: the slot size to average elements (< len(ts))
    :param a: the number of items of the language
    :param id: an id to concatenate to values of sax representation
    :return: the sax representation
    '''
    w = len(ts) // slot_size
    return sax(ts, w, a, id=id)
