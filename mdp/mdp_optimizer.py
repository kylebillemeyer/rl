import numpy as np

"""
mdp * [N x A] * int int -> [N x 1]: A model-free prediction algorithm that uses a
first pass monte carlo solution to estimate the value function of an
unknown. 

Note, this function cheats a bit and uses the underlying mdp structure to
perform the operation on all states. In a real model-free scenario we would
only know of states we've visited. The true purpose of this function would
be to estimate the value of a given state.
"""
def first_pass_monte_carlo(mdp, policy, num_trials, learning_rate):
	vpi = np.zeros(mdp.num_states)
	for i in range(mdp.num_states):
		state = mdp.states[i]

		for trial in range(num_trials):
			vpi[i] = temporal_difference(
				mdp, policy, state, vpi[i], learning_rate, -1)
	return vpi

"""
mdp * [N x A] * int -> [N x 1]: A model-free prediction algorithm that uses a
first pass monte carlo solution to estimate the value function of an
unknown. Every pass differs from frist pass in that each visit to the
state in question that occurs within a trials will weigh the average.
This trades off the number of trials needed to be run in a mdp with cycles
in order to have a useful estimation of the value function. However, doing
this introduces some bias to the evaluation because every cyclical path
implicitly has a non-cyclical subpath with will be weighed twice for every
trial.

Note, this function cheats a bit and uses the underlying mdp structure to
perform the operation on all states. In a real model-free scenario we would
only know of states we've visited. The true purpose of this function would
be to estimate the value of a given state.
"""
def every_pass_monte_carlo(mdp, policy, numTrials):
	vpi = np.zeros(mdp.num_states)
	for i in range(mdp.num_states):
		state = mdp.states[i]
		visits = 0
		running_average = 0
		
		for trial in range(numTrials):
			totalRewards = 0
			path = mdp.sample_path(state, policy)
			for node in reversed(path):
				totalRewards += node[1]
				if node[0] == state:
					running_average = ((running_average * visits) + totalRewards) / (visits + 1)
					visits += 1
		vpi[i] = running_average
	return vpi

"""
mdp * [A x N] * state * float * float * int: Performs a single iteration
temporal difference learning. Effectively, it provides a directional update
of the value estimation for a given state. More specifically, the function
estimates the the "target" value of the current state by running a trial with
fixed look ahead (the number of actions to be taken is defined by
learning_depth). It the determines the value error as the difference between
the target value and the expected value. The expected value is updated in
the direction of the error by a factor determined by learning_rate.

Note, setting learning_depth to -1 will effectively make this a single trial
monte-carlo simulation. Here, -1 means run until the look ahead terminates.
"""
def temporal_difference(mdp, policy, state, expected_value, learning_rate, learning_depth):
	td_target = mdp.sample(state, policy, learning_depth)
	td_error = td_target - expected_value
	return expected_value + (learning_rate * td_error)	

"""
mdp * [N x A] * int -> [N x 1]: Calculate the value function for a given policy function
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
	num_states = len(mdp.states)
	num_actions = len(mdp.actions)

	v = np.zeros(num_states)
	for i in range(iterations):
		vi = np.zeros(num_states)
		for j in range(num_actions):
			vi += policy[j, :] * (mdp.rewards[j] + (mdp.discount * mdp.probabilityMat[j].dot(v)))
		v = vi
	return v

"""
mdp * [N x A] * int -> [N x 1]: Calculates the optimal policy function for a given
mdp. It does this by doing a single dynamic update of the estimated value 
function for a starting policy function, then updates the policy function
using a greedy policy, i.e. a policy that says to always choose the action 
with the highest estimated expected reward. It can be proven that applying
this two step process iteratively is guaranteed to converge on the optimal
policy function.
"""
def optimized_policy_func(mdp, policy, iterations):
	num_states = len(mdp.states)
	num_actions = len(mdp.actions)

	v = np.zeros(num_states)
	for i in range(iterations):
		vi = np.zeros(num_states)
		for j in range(num_actions):
			vi += policy[j, :] * (mdp.rewards[j] + (mdp.discount * mdp.probabilityMat[j].dot(v)))
		v = vi
		policy = greedy_policy(v, mdp.encoded, mdp.num_states, mdp.num_actions)
		#print v
		#print policy
	return (v, policy)

"""
[N x 1] * [Transition] * int:A * int:N -> [A x N]: Given a value function,
builds the policy function that would always choose the action with the
highest estimated reward.
"""
def greedy_policy(vpi, transitions, num_states, num_actions):
	best_actions = {} # {from_state, (action, expected_reward)}

	#print transitions
	# Go through each transition and update the cache if this action has a higher
	# expected reward.
	for t in transitions:
		from_state = t[0]
		to_state = t[2]
		action = t[1]
		expected_reward = vpi[to_state]
		best_action = best_actions.setdefault(from_state, (action, expected_reward))

		#print "{}, {}, {}, {}".format(from_state, action, to_state, expected_reward)

		if best_action[1] <= expected_reward:
			best_actions[from_state] = (action, expected_reward)

	#print best_actions

	# build the policy matrix based on each states best action
	policy = np.zeros((num_actions, num_states))
	for state in range(num_states):
		best_action = best_actions[state]
		#print best_action
		if best_action is None:
			policy[0, state] = 1
		else:
			policy[best_action[0], state] = 1

	return policy
