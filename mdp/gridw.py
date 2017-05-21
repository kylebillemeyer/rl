import mdp as m

def create_grid(x, y, terminals, discount):
	transitions = []
	states = []

	for j in range(y):
		for i in range(x):
			name = _name(i, j)
			states.append(name)

			reward = -1
			if ((i, j) in terminals):
				reward = 0

			transitions.append((name, 'n', _name(i, max(j-1, 0)), 1, reward))
			transitions.append((name, 'e', _name(min(i+1, x-1), j), 1, reward))
			transitions.append((name, 's', _name(i, min(j+1, y-1)), 1, reward))
			transitions.append((name, 'w', _name(max(i-1, 0), j), 1, reward))

	return m.MDP(
		states,
		['n', 'e', 's', 'w'],
		transitions,
		{_name(t[0], t[1]) for t in terminals},
		discount
	)

def _name(x, y):
	return "({},{})".format(x, y)