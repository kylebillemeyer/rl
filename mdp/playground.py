import mdp
import utils
import mdp_optimizer as mdpo
import numpy as np
import gridw

transitions = [
    ('c1', 'facebook', 'distraction', 1.0, -1),
    ('c1', 'study', 'c2', 1.0, -2),
    ('distraction', 'facebook', 'distraction', 1.0, -1),
    ('distraction', 'quit', 'c1', 1.0, 0),
    ('c2', 'study', 'c3', 1.0, -2),
    ('c2', 'sleep', 'rest', 1.0, 0),
    ('c3', 'study', 'rest', 1.0, 10),
    ('c3', 'pub', 'c1', 0.2, 1),
    ('c3', 'pub', 'c2', 0.4, 1),
    ('c3', 'pub', 'c3', 0.4, 1)
]

test_mdp = mdp.MDP(
    ['c1', 'distraction', 'c2', 'c3', 'rest'],
    ['facebook', 'quit', 'study', 'sleep', 'pub'],
    transitions,
    {'rest'},
    1)

test_policy = np.array([
    (0.5, 0.0, 0.5, 0.0, 0.0),
    (0.5, 0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.5, 0.0),
    (0.0, 0.0, 0.5, 0.0, 0.5),
    (0.0, 0.0, 0.0, 1.0, 0.0),
], dtype = float)

#print test_mdp.sample('c2', test_policy, True)

"""
mdpo.calc_expected_value_by_sampling(test_mdp, 'c1', 10000, test_policy)
mdpo.calc_expected_value_by_sampling(test_mdp, 'distraction', 10000, test_policy)
mdpo.calc_expected_value_by_sampling(test_mdp, 'c2', 10000, test_policy)
mdpo.calc_expected_value_by_sampling(test_mdp, 'c3', 10000, test_policy)
mdpo.calc_expected_value_by_sampling(test_mdp, 'rest', 10000, test_policy)
"""
"""
print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 1)
print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 10)
print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 100)
print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 1000)
print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 10000)
"""

gw = gridw.create_grid(4, 4, [(0, 0), (3, 3)], 1)

gw_policy = np.array([
    (0, 0, 0, 0),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (.25, .25, .25, .25),
    (0, 0, 0, 0),
], dtype = float)
#print gw.states
#print gw.actions
#print gw.terminals
#print gw.probabilityMat
#print gw.rewards
print mdpo.calc_value_func_dynamic(gw, gw_policy, 1)
