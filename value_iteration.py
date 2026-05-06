import numpy as np
from mdps import MDP


def one_step_lookahead(mdp, V, state, gamma):
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for action in range(mdp.n_actions):
        for next_state in range(mdp.n_states):
            prob = mdp.get_transition_probability(state, action, next_state)
            reward = mdp.get_reward(state, action)
            Q[state, action] += prob * (reward + gamma * V[next_state])
    return Q


def value_iteration(mdp: MDP, gamma: float = 0.95, theta: float = 0.01):
    V = np.zeros([mdp.n_states])
    delta = 1
    while theta < delta:
        delta = 0
        for state in range(mdp.n_states):
            Q = one_step_lookahead(mdp, V, state, gamma)
            update = np.max(Q[state])

            delta = max(delta, np.abs(update - V[state]))
            V[state] = update

    policy = np.zeros([mdp.n_states, mdp.n_actions])
    for state in range(mdp.n_states):
        Q = one_step_lookahead(mdp, V, state, gamma)
        best_action = np.argmax(Q[state])
        policy[state, best_action] = 1

    return V, policy
