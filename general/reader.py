import csv
import datetime
from collections import defaultdict


def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            line_split = line.split(',')
            sequence = [line_split[0]]
            sequence += line_split[2].strip()

            sequences.append(sequence)

    for sequence in sequences:
        for itemset_i in range(1, len(sequence)):
            sequence[itemset_i] = set(sequence[itemset_i])

    return sequences


def read_data_kosarak(filename):
    """
    :param filename:
    :return: [[class, {}, {}, ...], [class, {}, {}, ...]]
    """
    with open(filename) as f:
        data = transform_kosarak(f)
    return data


def transform_kosarak(iterator):
    data = []
    for line in iterator:
        line_split = line.split("-1")
        if len(line_split) > 1:
            split_first_itemset = line_split[0].split()
            first = split_first_itemset[0]

            second = set(split_first_itemset[1:])
            line_split = line_split[1:-1]

            sequence = [first, second]

            for itemset in line_split:
                items = itemset.split()
                new_itemset = set(items)
                sequence.append(new_itemset)

        data.append(sequence)
    return data


def read_data_sc2(filename):
    """
    :param filename:
    :return: [[class, {}, {}, ...], [class, {}, {}, ...]]
    """
    data = []
    with open(filename) as f:
        for line in f:
            sequence = []
            sequence.append(line[-8])
            line = line[:-8]

            line_split = line.split("-1")[:-2]

            for itemset in line_split:
                items = itemset.split()
                new_itemset = set(items)
                sequence.append(new_itemset)

            if len(sequence) > 1:
                data.append(sequence)
    return data


def read_jmlr(target_word, path):
    return_sequences = []

    with open('{}.lab'.format(path)) as dict_files:
        data_dict = {}
        for i, line in enumerate(dict_files):
            data_dict[i] = line[:-1]

    with open('{}.dat'.format(path)) as jmlr:
        data = jmlr.readline()
        data = data.split("-1")

        for seq in data:
            sequence = [set([data_dict[int(i)]]) for i in seq.split(" ") if i != '']
            return_sequences.append(sequence)

    # now we add the class for the presence of a word, and we remove the target word
    for sequence in return_sequences:
        if set([target_word]) in sequence:
            sequence.insert(0, '+')

            # now we remove the element !
            while "Removing the element":
                try:
                    sequence.remove(set([target_word]))
                except ValueError:
                    break
        else:
            sequence.insert(0, '-')

    return return_sequences


def read_retail(target_item_id, path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        items_dict = {}
        items_histo = defaultdict(int)

        # contains itemset-date
        client_sequences = defaultdict(list)

        # we skip first line
        next(reader)

        for row in reader:
            if len(row) != 9:
                # we skip non-conventional rows
                continue

            item, date_i, description, client_id = row[1], row[4], row[2], row[-2]

            try:
                date_i = datetime.strptime(date_i, '%d/%m/%Y %H:%M')
            except ValueError:
                # we have a bad formated line, we do not take her
                continue
            items_histo[item] += 1
            items_dict[item] = description
            client_seq = client_sequences[client_id]
            item_added = False

            # we check if there is already an itemset for this day
            for itemset, date_itemset in client_seq:
                if date_i == date_itemset:
                    itemset.add(item)
                    item_added = True

            if not item_added:
                client_seq.append([{item}, date_i])

    output = []
    # we need to sort client sequences according to dates
    count = 0
    for client, sequence in client_sequences.items():
        seq_output = []
        sorted_seq = sorted(sequence, key=lambda x: x[1])
        seq_class = '-'

        # now we take the targeted item, and we remove itemsets after it: we are looking for previous pattern predicting
        # the buying
        for itemset, time_itemset in sorted_seq:
            if target_item_id in itemset:
                count += 1
                seq_class = '+'
                itemset.remove(target_item_id)
                if len(itemset) > 0:
                    seq_output.append(itemset)
                break

            seq_output.append(itemset)

        seq_output.insert(0, seq_class)

        if len(seq_output) > 1:
            output.append(seq_output)

    return output, items_dict


def read_data_rotten(filename):
    data = []
    with open(filename) as f:
        for line in f:
            sequence = []
            class_line = line.split(',')[0]

            # we remove \n and ., spaces at the end and begining, and we split
            sequence_line = line[len(class_line) + 1:-2].strip(' ').split(' ')
            sequence.append(class_line)

            for word in sequence_line:
                itemset = set()
                itemset.add(word)
                sequence.append(itemset)

            if len(sequence) > 1:
                data.append(sequence)
    return data
