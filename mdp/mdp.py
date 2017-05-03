import numpy as np
import itertools as it
import random

class MDP():
    """Represents a Markov Decision Process"""

    """
    states: [String]
    actions: [String]
    transitions: [(state, action, state, probability, reward))]
    terminals: {state}
    discount: num between 0 and 1
    """
    def __init__(self, states, actions, transitions, terminals, discount):
        self.states = states
        self.actions = actions
        self.terminals = terminals

        self.state_e = {states[i]: i for i in range(len(states))}
        self.action_e = {actions[i]: i for i in range(len(actions))}
        encoded = self.endcodeTransistions(states, actions, transitions)

        num_states = len(states)
        self.probabilityMat = [self.generateProbabilities(encoded, num_states, self.action_e[action]) for action in actions]
        self.rewards = [self.generateRewards(encoded, num_states, self.action_e[action]) for action in actions]
        self.discount = discount

    """
    Encodes each transition by mapping labels to their array index position.
    This allows internal matrix lookups to be done by the encoded value.

    states: [String]
    actions: [String]
    transitions: [(state, action, state, probability, reward))]

    returns: [(state_encoded, action_encoded, state_encoded, probability, reward)]
    """
    def endcodeTransistions(self, states, actions, transitions):
        return [self._enc(t) for t in transitions]

    def _enc(self, t):
        return (self.state_e[t[0]], self.action_e[t[1]], self.state_e[t[2]], t[3], t[4])

    """
    Generates an N x N matrix where N is the number of states.
    Each entry i, j represents the probability of transitioning
    from state i to state j when taking the given action.

    transitions: [(state, action, state, probability, reward))]
    num_states: int
    action: String

    returns: [N x N]
    """
    def generateProbabilities(self, transitions, num_states, action):
        mat = np.zeros((num_states, num_states))

        for t in transitions:
            if t[1] == action:
                from_state = t[0]
                to_state = t[2]
                probability = t[3]
                mat[from_state, to_state] = probability
        return mat

    """
    Generates an N x 1 matrix where N is the number of states.
    Each entry i represents the reward for leaving the current
    state when taking the given action.

    transitions: [(state, action, state, probability, reward))]
    num_states: int
    action: String

    returns: [N x 1]
    """
    def generateRewards(self, transitions, num_states, action):
        mat = np.zeros(num_states)

        for t in transitions:
            if t[1] == action:
                from_state = t[0]
                reward = t[4]
                mat[from_state] = reward
        return mat

    def sample(self, root, policy, debug=False):
        state = self.state_e[root]
        prev = state
        discount = 1
        totalReward = 0

        # performance tweak to do integer equality instead of string
        encodedTerminals = {self.state_e[t] for t in self.terminals}

        while(state not in encodedTerminals):
            prev = state
            choices = policy[state]
            # take a random action based on the policy probabilities
            choice = self._randomArr(choices)
            # transition state based on the probability for taking the
            # selected action at the current state
            state = self._randomArr(self.probabilityMat[choice][state])
            # collect reward for the action
            reward = self.rewards[choice][prev] * discount
            totalReward += reward
            discount *= self.discount

            if debug:
                print "In state {}, took action {} and moved to state {}, collecting {} reward".format(
                    self.states[prev], self.actions[choice], self.states[state], reward
                )

        return totalReward

    def _randomArr(_, arr):
        if (sum(arr) != 1.0):
            raise ValueError('array distribution does not sum to 1')

        cum_prob = np.cumsum(arr)
        rand = random.random()

        for i in range(len(cum_prob)):
            if (rand < cum_prob[i]):
                return i

        raise ValueError('malformed array distribution')

    def averageExpectedReward(self, state, numTrials, policy):
        totalRewards = 0
        for i in range(numTrials):
            totalRewards += self.sample(state, policy)
        print "average expected reward for {}: {}".format(state, totalRewards / numTrials)

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

test_mdp = MDP(
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
test_mdp.averageExpectedReward('c1', 10000, test_policy)
test_mdp.averageExpectedReward('distraction', 10000, test_policy)
test_mdp.averageExpectedReward('c2', 10000, test_policy)
test_mdp.averageExpectedReward('c3', 10000, test_policy)
test_mdp.averageExpectedReward('rest', 10000, test_policy)


