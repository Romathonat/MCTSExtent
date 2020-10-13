import random

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from mctsextent.main import launch_mcts
from general.utils import is_subsequence

from general.reader import transform_kosarak
from sax.sax_main import sax_slot_size, readable_pattern


def read_energy_data():
    # df = pd.read_csv('../data/energydata.csv', index_col='date')
    df = pd.read_csv('../data/energydata.csv')

    # we need to find failures, and to get the timestamp. In our case, this corresponds to cases where appliances are too high:
    # let's say that we label when applicances are > 800

    timefails = df[['Appliances']][df.Appliances > 700]
    timeseries = df.loc[:, 'lights':]

    return timeseries, timefails

def format_sequence(sequence, class_name):
    """
    Format sequences to give them to mctsextent
    :param sequences: a list of sequences in pandas format
    :param class_name: the class name
    :return:
    """
    formated_sequence = [class_name]

    for item in sequence:
        formated_sequence.append({item})

    return formated_sequence


def format_sequences(sequences, class_name):
    """
    Format sequences to give them to mctsextent
    :param sequences: a list of sequences in pandas format
    :param class_name: the class name
    :return:
    """
    return_sequences = []

    for sequence in sequences:
        formated_sequence = [class_name]
        for _, itemset in sequence.iterrows():
            itemset = itemset.values
            itemset_format = set()

            for item in itemset:
                itemset_format.add(item)
            formated_sequence.append(itemset_format)
        return_sequences.append(formated_sequence)

    return return_sequences


def sax_to_fail_sequences(df_sax, timefails, nb_events_back):
    '''
    Find fails, take events before. This function also extracts as many sequences with no fail, to balance data.

    :param df_sax:
    :param timefails: pandas serie with fails in the order of time
    :return: sequences to be ingested by discriminative algorithm
    '''

    current_index_fail = 0
    # the fail index in paa discretization
    current_fail = timefails.iloc[current_index_fail]['index_paa']

    fail_sequences = []
    normal_sequences = []
    # print("The size of the paa rpz is: {}".format(len(df_sax)))

    for i in range(df_sax.shape[0]):
        # print('i: {}, index: {}'.format(i, row.Index))
        if i > current_fail:
            # we just passed the fail
            # we take points before
            fail_sequences.append(df_sax.iloc[i - nb_events_back:i])

            current_index_fail += 1
            # we stop if we found all fails
            if current_index_fail >= len(timefails):
                break

            current_fail = timefails.iloc[current_index_fail]['index_paa']

        else:
            if current_index_fail > 0:
                previous_fail = timefails.iloc[current_index_fail - 1]['index_paa']
            else:
                previous_fail = 0

            # the first condition tests if previous window is not in the previous "window fail zone"
            # the second condition makes sure we are not in a "window fail zone"
            if i - nb_events_back > previous_fail and i < current_fail - nb_events_back:
                normal_sequences.append(df_sax.iloc[i - nb_events_back:i])
                # count_without_fail = 0

    # we add random normal sequences
    all_sequences = format_sequences(fail_sequences, '+')

    # should do it with a proper python exception but no time
    if len(fail_sequences) > len(normal_sequences):
        print("ERROR: Number of normal sequences is too LOW !")

    print("number of positive sequences: {}, number of negative sequences: {}".format(len(fail_sequences),
                                                                                      len(normal_sequences)))

    all_sequences.extend(format_sequences(random.sample(normal_sequences, len(fail_sequences)), '-'))
    # all_sequences.extend(random.sample(format_sequences(normal_sequences, '-'), 5 * len(timefails)))
    # now we can remove the time, and return the good format

    return all_sequences


def discretize_timeseries(timeseries, nb_points_slot, a):
    # we iterate through timeseries to have their sax representation
    df_sax = None
    means = []
    stds = []
    lookup = []
    id_to_column_name = {}

    for i, ts in enumerate(timeseries):
        id_to_column_name[i] = timeseries.columns[i]
        sax_rpz, mean, std, lookup = sax_slot_size(timeseries[ts], nb_points_slot, a, id=i)
        means.append(mean)
        stds.append(std)
        if df_sax is None:
            df_sax = sax_rpz
        else:
            df_sax = pd.concat([df_sax, sax_rpz], axis=1)

    return df_sax, means, stds, id_to_column_name, lookup


def discretize_and_label_sequences(timeseries, timefails, nb_points_slot, a, nb_events_back):
    """
    :param timeseries:
    :param timefails:
    :param nb_points_slot: number of point for each sloint
    :param a: number of discretization slot
    :param nb_events_back:
    :return: discretized sequences, looup_table, means, stds, and id_to_column_name to reconstruct correct pattern, timefails with the right index (considering discretization)
    """
    df_sax, means, stds, id_to_column_name, lookup = discretize_timeseries(timeseries, nb_points_slot, a)
    # print(df_sax)

    # we capture the moment where we have a fail: we go back in time, tacking nb_point_back before the fail
    # we add a column to timefails representing the index where fails appear, considering the paa
    timefails['index_paa'] = (timefails.index / nb_points_slot).astype(int)
    timefails.reset_index(drop=True, inplace=True)

    # we remove duplicates: if event appeared several times in the same paa segment, it is one event
    timefails.drop_duplicates(subset='index_paa', keep='first', inplace=True)

    sequences = sax_to_fail_sequences(df_sax, timefails, nb_events_back)

    return sequences, lookup, means, stds, id_to_column_name,


def ts_mining(sequences, lookup, means, stds, id_to_column_name):
    '''
    Mine discriminative patterns of interval values before the fail appears.
    For now, we consider taking 100 points before the fail.
    We consider timeseries to be sampled equally, at the same time.
    :param sequences: a list of distretized sequences, with labels
    :return: the patterns
    '''
    # we mine them with mctsextent
    patterns = launch_mcts(sequences, '+', top_k=100, time_budget=10, iterations_limit=2 ** 30)

    # we convert obtained pattern to original form to give to experts
    # for pattern in patterns:
    #    print('WRAcc {}: {}'.format(pattern[0], readable_pattern(pattern[1], lookup, means, stds,
    #                                                             id_to_column_name=id_to_column_name)))

    return patterns


# TODO: check ecg https://www.kaggle.com/c/seizure-prediction/data
# https://www.kaggle.com/c/belkin-energy-disaggregation-competition/data
def sequence_to_xgboost(sequences):
    set_of_itemset = set()

    for sequence in sequences:
        for itemset in sequence:
            set_of_itemset.add(frozenset(itemset))

    relabeled_sequences = []
    for sequence in sequences:
        new_sequence = [sequence[0]] + [0 for i in range(len(set_of_itemset))]
        for i, itemset in enumerate(set_of_itemset):
            if itemset in sequence:
                new_sequence[i + 1] = 1

        relabeled_sequences.append(new_sequence)

    return relabeled_sequences


def dataset_to_pattern_features(sequences, patterns):
    out = []
    for sequence in sequences:
        relabeled_sequence = [sequence[0]] + [0 for i in range(len(patterns))]

        for i, pattern in enumerate(patterns):
            if is_subsequence(pattern[1], sequence[1:]):
                relabeled_sequence[i + 1] = 1

        out.append(relabeled_sequence)

    return out


def predict_worms():
    sequences_train = []

    with open('../data/Worms_TRAIN.txt') as f:
        for line in f:
            sequences_train.append(line.split())

    df = pd.DataFrame(sequences_train)
    df = df.transpose().astype(float)

    sequences_train = []
    for column in df:
        # we do not take the first element
        ts = df.loc[1:, column].reset_index(drop=True)
        sax_rpz, means, stds, lookup = sax_slot_size(ts, 10, 10)

        sequences_train.append(format_sequence(sax_rpz, str(df.loc[0, column])))

    patterns = launch_mcts(sequences_train, '1.0', top_k=100, time_budget=10, iterations_limit=2 ** 30)
    patterns2 = launch_mcts(sequences_train, '2.0', top_k=100, time_budget=10, iterations_limit=2 ** 30)
    patterns3 = launch_mcts(sequences_train, '3.0', top_k=100, time_budget=10, iterations_limit=2 ** 30)
    patterns4 = launch_mcts(sequences_train, '4.0', top_k=100, time_budget=10, iterations_limit=2 ** 30)
    patterns5 = launch_mcts(sequences_train, '5.0', top_k=100, time_budget=10, iterations_limit=2 ** 30)

    patterns = patterns + patterns2 + patterns3 + patterns4 + patterns5

    sequences_test = []

    with open('../data/Worms_TEST.txt') as f:
        for line in f:
            sequences_test.append(line.split())

    df = pd.DataFrame(sequences_test)
    df = df.transpose().astype(float)

    sequences_test = []
    for column in df:
        # we do not take the first element
        ts = df.loc[1:, column].reset_index(drop=True)
        sax_rpz, means, stds, lookup = sax_slot_size(ts, 1, 10)

        sequences_test.append(format_sequence(sax_rpz, str(df.loc[0, column])))

    relabeled_train_sequences = dataset_to_pattern_features(sequences_train, patterns)
    relabeled_test_sequences = dataset_to_pattern_features(sequences_test, patterns)

    df_train = pd.DataFrame(relabeled_train_sequences, columns=['class'] + [str(pattern[1]) for pattern in patterns])
    df_test = pd.DataFrame(relabeled_test_sequences, columns=['class'] + [str(pattern[1]) for pattern in patterns])

    y_train, X_train = df_train.iloc[:, 0], df_train.iloc[:, 1:]
    y_test, X_test = df_test.iloc[:, 0], df_test.iloc[:, 1:]

    xg_classif = xgb.XGBClassifier()
    xg_classif.fit(X_train, y_train)
    preds = xg_classif.predict(X_test)

    print("XGBoost accuracy: {}".format(accuracy_score(y_test, preds)))

def predict_consumption_energy():
    timeseries, timefails = read_energy_data()

    sequences, lookup, means, stds, id_to_column_name = discretize_and_label_sequences(timeseries, timefails, 2, 48, 5)
    random.shuffle(sequences)

    skf = StratifiedKFold(n_splits=6)

    for train_index, test_index in skf.split([sequence[1:] for sequence in sequences],
                                             [sequence[0] for sequence in sequences]):
        train_sequences = []
        test_sequences = []

        for i in train_index:
            train_sequences.append(sequences[i])

        for i in test_index:
            test_sequences.append(sequences[i])

        patterns = ts_mining(train_sequences, lookup, means, stds, id_to_column_name)

        # relabelize dataset
        relabeled_train_sequences = dataset_to_pattern_features(train_sequences, patterns)
        relabeled_test_sequences = dataset_to_pattern_features(test_sequences, patterns)

        df_train = pd.DataFrame(relabeled_train_sequences,
                                columns=['class'] + [str(pattern[1]) for pattern in patterns])
        df_test = pd.DataFrame(relabeled_test_sequences, columns=['class'] + [str(pattern[1]) for pattern in patterns])

        # test naive baseline
        # relabeled_dataset = sequence_to_xgboost(sequences)
        # df = pd.DataFrame(relabeled_dataset )

        y_train, X_train = df_train.iloc[:, 0], df_train.iloc[:, 1:]
        y_test, X_test = df_test.iloc[:, 0], df_test.iloc[:, 1:]

        # classical
        xg_classif = xgb.XGBClassifier()
        xg_classif.fit(X_train, y_train)
        preds = xg_classif.predict(X_test)

        print("XGBoost accuracy: {}".format(accuracy_score(y_test, preds)))
        print("Xgboost Precision: {}, Recall: {}".format(precision_score(y_test, preds, pos_label='+'),
                                                         recall_score(y_test, preds, pos_label='+')))

        '''
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        print("5NN Precision: {}, Recall: {}".format(precision_score(y_test, preds, pos_label='+'),
                                                 recall_score(y_test, preds, pos_label='+')))
    '''

    # k-fold
    # kf = KFold(n_splits=8)
    #
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #
    #     xg_classif = xgb.XGBClassifier()
    #     xg_classif.fit(X_train, y_train)
    #
    #     preds = xg_classif.predict(X_test)
    #
    #     print('Len train {}, len test {}'.format(len(X_train), len(X_test)))
    #     print(accuracy_score(y_test, preds))


# predict_consumption_energy()
print(predict_worms())
'''
Number iteration mcts: 124522
WRAcc 0.08606557377049183: [{'lights': [-0.35965568545824667, 1.7913663121883836]}, {'lights': [13.971994497978196, 'inf']}]
WRAcc 0.08606557377049179: [{'RH_3': [38.417983441364626, 39.242500077205484]}]
WRAcc 0.07377049180327869: [{'T2': [19.785650048150526, 20.341219463847917]}, {'RH_out': [75.97536631994986, 79.75041803901698]}]
WRAcc 0.06967213114754099: [{'rv1': [21.315446342387737, 24.988033485049435], 'rv2': [21.315446342387737, 24.988033485049435]}, {'RH_2': [42.55457812411228, 43.84557428244676]}]
WRAcc 0.06967213114754098: [{'lights': [-0.35965568545824667, 1.7913663121883836]}, {'rv2': [24.988033485049435, 28.660620627711136]}, {'lights': [13.971994497978196, 'inf']}]
'''
