from scipy.stats import norm


def lookup_table(x):
    breaks = []
    slot_sum = 1 / x

    for i in range(1, x):
        breaks.append(norm.ppf(slot_sum * i))

    return breaks

def lookup_table_equal_frequency(x, ts):
    """
    create a lookup table of x slot, with the same number of element in each slot
    :param x:
    :param ts:
    :return:
    """
    breaks = []
    ts = ts.sort_values()
    slot_size = len(ts) // x
    for i in range(1, x):
        breaks.append(ts.iloc[i * slot_size])

    return breaks
