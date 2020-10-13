import pathlib

from mctsextent.main import launch_mcts
from general.reader import read_data_kosarak

def test_mcts_simple():
    DATA = read_data_kosarak(pathlib.Path(__file__).parent.parent / 'data/easy.data')
    results = launch_mcts(DATA, '+', time_budget=12000, top_k=10, iterations_limit=10000, theta=1)

    assert len(results) >= 4

    a = (frozenset(['1']), frozenset(['1']), frozenset(['5']), frozenset(['5']), frozenset(['5']), frozenset(['5']), frozenset(['5']))
    b = (frozenset(['1']), frozenset(['4']), frozenset(['5']))
    c = (frozenset(['1']), frozenset(['5']))

    check_element = []

    for result in results:
        check_element.append(result[1])

    assert a in check_element
    assert b in check_element
    assert c in check_element
