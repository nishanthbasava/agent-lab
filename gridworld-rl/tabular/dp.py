import numpy as np


def policy_evaluation(P, policy, gamma=0.99, theta=1e-8):
    #Iterative policy evaluation via the Bellman expectation backup.
    #policy is a [n_states, 4] array of action probabilities.
    #Returns V as [n_states].

    n_states = len(P)
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            v_new = 0.0
            for a in range(4):
                for prob, s2, r, done in P[s][a]:
                    # V(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a) [r + gamma * V(s')]
                    # done zeroes the bootstrap: no future value past a terminal
                    v_new += policy[s, a] * prob * (r + gamma * V[s2] * (not done))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            return V


if __name__ == "__main__":
    from env.gridworld import GridWorld, GRID

    env = GridWorld(GRID)
    uniform = np.full((env.n_states, 4), 0.25)
    V = policy_evaluation(env.P, uniform)
    print(V.reshape(env.nrows, env.ncols).round(2))
