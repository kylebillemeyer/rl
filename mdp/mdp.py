import numpy as np
import itertools as it
import utils

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
        self.encoded = self.endcodeTransistions(states, actions, transitions)

        self.num_states = len(states)
        self.num_actions = len(actions)
        self.probabilityMat = [self.generateProbabilities(self.encoded, self.num_states, self.action_e[action]) for action in actions]
        self.rewards = [self.generateRewards(self.encoded, self.num_states, self.action_e[action]) for action in actions]
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

    def sample_path(self, root, policy, depth=-1, debug=False):
        state = self.state_e[root]
        prev = state
        discount = 1
        count = 0

        # performance tweak to do integer equality instead of string
        encodedTerminals = {self.state_e[t] for t in self.terminals}
        path = []
        while(state not in encodedTerminals and (depth == -1 or count < depth)):
            prev = state
            choices = policy[state]
            # take a random action based on the policy probabilities
            choice = utils.randomArr(choices)
            # transition state based on the probability for taking the
            # selected action at the current state
            state = utils.randomArr(self.probabilityMat[choice][state])
            # collect reward for the action
            reward = self.rewards[choice][prev] * discount
            path.append((self.states[state], reward))
            discount *= self.discount
            count += 1

            if debug:
                print "In state {}, took action {} and moved to state {}, collecting {} reward".format(
                    self.states[prev], self.actions[choice], self.states[state], reward
                )

        return path

    def sample(self, root, policy, depth=-1, debug=False):
        path = self.sample_path(root, policy, depth, debug)
        if not path:
            return 0
        
        return reduce(lambda x, y: x + y, map(lambda x: x[1], path))

