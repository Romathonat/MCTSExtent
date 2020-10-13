from mctsextent.node import Node

def test_node_creation():
     data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]
     intent = [{'A'}, {'C'}]
     node = Node(intent, None, data, [data[0]], '+', {})

     assert node.quality == 0.25
     assert len(node.extend_positive) == 1
     assert len(node.candidate_sequences_expand) == 0
     assert node.get_normalized_quality() == 1

def test_node2():
     data = [['+', {'A', 'B'}, {'C'}], ['+', {'A'}, {'D'}], ['-', {'A'}, {'B'}], ['-', {'A'}, {'B'}, {'D'}]]
     intent = [{'A'}, {'C'}]
     node = Node(intent, None, data, [data[0], data[1]], '+', {})

     assert node.quality == 0.125
     assert len(node.extend_positive) == 1
     assert len(node.candidate_sequences_expand) == 1

def test_expanded():
     data = [['+', {'A', 'B'}, {'C'}], ['+', {'A'}, {'D'}], ['-', {'A'}, {'B'}], ['-', {'A'}, {'B'}, {'D'}]]
     intent = [{'A'}, {'C'}]
     data_positive = [data[0], data[1]]

     node = Node(intent, None, data, data_positive, '+', {})

     assert not node.is_fully_expanded()

     expanded_node = node.expand(data, data_positive, '+')

     assert expanded_node.intent == (frozenset({'A'},),)
     assert len(node.candidate_sequences_expand) == 0
     assert node.is_fully_expanded()
     assert expanded_node.is_terminal()

def test_update():
     data = [['+', {'A', 'B'}, {'C'}], ['+', {'A'}, {'D'}], ['-', {'A'}, {'B'}], ['-', {'A'}, {'B'}, {'D'}]]
     intent = [{'A'}, {'C'}]
     data_positive = [data[0], data[1]]

     node = Node(intent, None, data, data_positive, '+', {})

     assert node.quality == 0.125

     node.update(0.075)

     assert node.quality == 0.100
     assert node.number_visits == 2


