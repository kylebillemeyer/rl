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

print mdpo.first_pass_monte_carlo(test_mdp, test_policy, 10000, .01)
#print mdpo.every_pass_monte_carlo(test_mdp, test_policy, 10000)
#print mdpo.calc_value_func_dynamic(test_mdp, test_policy, 1000)
print mdpo.temporal_difference(test_mdp, test_policy, 'c1', 0, .01, -1)

width = 5
height = 10
gw = gridw.create_grid(width, height, [(0, 0), (width-1, height-1)], 1)
gw_policy = np.zeros((4, height*width))
gw_policy.fill(.25)
gw_policy[:,0] = np.zeros((1, 4))
gw_policy[:,height*width-1] = np.zeros((1, 4))

#print gw_policy

#print mdpo.calc_value_func_dynamic(gw, gw_policy, 1000)
pstar = mdpo.optimized_policy_func(gw, gw_policy, 1000)
#print pstar[0]
#print pstar[1]
