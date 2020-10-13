# MCTSExtent

This repo holds the code sent to KAIS. We will open it in free access on Github once we have published. 

**MCTSExtent** finds subgroups for sequential datasets. More precisely, you give a dataset of sequences of itemsets with each sequence having a class, a target class,
and the algorithm finds the top-k best pattern discriminative of this class.

This is interesting for two reasons:
1. You can use patterns found to better predict classes of sequences
2. You can use patterns to better understand a phenomena (= what are the sequence of events, i.e the **patterns**, that appears for this target class)

To use this module, you need to have data in a kosarak-like format. For exemple, the sequence "{1 5},{5 8 9}, {2}", with a class of "+" is encoded this way:
```
+ 1 5 -1 5 8 9 -1 2 -1 -2
```
Each line then corresponds to a new sequence.

You also need to specify a target class. In this case, you could launch the algorithm this way:
``` python
from mctsextent.mctsextent.main import get_patterns

get_patterns(path='my_path', target_class='+')
```
Which would give you, by default, the top-5 non redundant patterns for target class '+'

In the following code we specify the number of patterns we want to get, and the value of theta for non-redundancy.

``` python
from mctsextent.mctsextent.main import get_patterns
from mctsextent.utils import print_results

results = get_patterns(path='../data/figures_rc.dat', target_class='3', time=10, top_k=10, theta=0.5)
```

### Code organization:
* Main code is present in mctsextent
* Tests contains unit test.
* Competitors holds the code of beam_search, misere, and an exhaustive one that we created to access the ground truth (see experiments of paper)

### Experiments
xp folder hold the code for experiments.
For reproducibility, you can relaunch experiments very easily by doing a:

```bash
pip install -r requirements.txt
cd xp
python3 xp_main.py
```
