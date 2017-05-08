import numpy as np

def calc_expected_value_by_sampling(mdp, state, numTrials, policy):
    totalRewards = 0
    for i in range(numTrials):
        totalRewards += mdp.sample(state, policy)
    print "average expected reward for {}: {}".format(state, totalRewards / numTrials)

"""
mdp * [N x A] -> [N x 1]: Calculate the value function for a given policy function
using a dynamic programming methodology. 

The policy function is an N x A matrix where N is the number of 
states and A is the number of actions that can be taken in general. 
Each value for index postion (n, a) is the probablity of taking 
that action. If a given action a is not valid for a given state n, 
then index [n, a] should be 0. Each row vector n must add up to 1.

The result is an N x 1 matrix where N is the number of states.
Each value for index n is the expected reward for state n.
"""    
def calc_value_func_dynamic(mdp, policy, iterations):
	# precompute the average award for each state
	avg_reward = calc_avg_rewards(mdp, policy)
	print avg_reward

	num_states = len(mdp.states)
	v = np.zeros(num_states)
	for i in range(iterations):
		#print "v: {}".format(v)
		#print "p: {}".format(policy)
		#print "p.v: {}".format(policy.dot(v))
		#print "v:{}, p:{}".format(v.shape, policy.shape)
		v = avg_reward + (mdp.discount * policy.dot(v))
	return v

"""
mdp * [N x A] -> [N x 1]: Calculates the average reward for each
state based on the weighted average of rewards gained per action a 
at a given state n.
"""
def calc_avg_rewards(mdp, policy):
    num_states = len(mdp.states)
    num_actions = len(mdp.actions)

    avg_rewards = np.zeros(num_states)
    for i in range(num_actions):
    	print policy[i,:]
    	print mdp.rewards[i]
        avg_rewards += policy[i,:] * mdp.rewards[i]
    return avg_rewards

