import mdp as m

def create_grid(x, y, terminals, discount):
	transitions = []
	states = []

	for i in range(x):
		for j in range(y):
			name = _name(i, j)
			states.append(name)
			if i > 0:
				transitions.append((name, 'w', _name(i-1, j), 1, -1))
			if i < x-1:
				transitions.append((name, 'e', _name(i+1, j), 1, -1))
			if j > 0:
				transitions.append((name, 's', _name(i, j-1), 1, -1))
			if j < y-1:
				transitions.append((name, 'n', _name(i, j+1), 1, -1))

	return m.MDP(
		states,
		['w', 'e', 's', 'n'],
		transitions,
		{_name(t[0], t[1]) for t in terminals},
		discount
	)

def _name(x, y):
	return "({},{})".format(x, y)