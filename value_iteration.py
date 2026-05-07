import numpy as np
from mdps import MDP, RiverSwimMDP


def one_step_lookahead(mdp, V, state, gamma):
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for action in range(mdp.n_actions):
        for next_state in range(mdp.n_states):
            prob = mdp.get_transition_probability(state, action, next_state)
            reward = mdp.get_reward(state, action)
            Q[state, action] += prob * (reward + gamma * V[next_state])
    return Q


def optim_one_step_lookahead(mdp, V, state, gamma):
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    for next_state in range(mdp.n_states):
        probs = mdp.P[state, :, next_state]
        rewards = mdp.R[state]
        Q[state, :] = probs * (rewards + gamma * V[next_state])
    return Q


def q_one_step_lookahead(mdp, Q, state, gamma):
    for action in range(mdp.n_actions):
        q_sum = 0
        for next_state in range(mdp.n_states):
            prob = mdp.get_transition_probability(state, action, next_state)
            reward = mdp.get_reward(state, action)
            q_sum += prob * (reward + gamma * np.max(Q[next_state]))
        Q[state, action] = q_sum
    return Q


def value_iteration_q(mdp: MDP, gamma: float = 0.95, theta: float = 0.01):
    V = np.zeros([mdp.n_states])
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    delta = 1
    while theta < delta:
        delta = 0
        for state in range(mdp.n_states):
            Q = q_one_step_lookahead(mdp, Q, state, gamma)
            update = np.max(Q[state])

            delta = max(delta, np.abs(update - V[state]))
            V[state] = update

    policy = np.zeros([mdp.n_states, mdp.n_actions])
    for state in range(mdp.n_states):
        Q = one_step_lookahead(mdp, V, state, gamma)
        best_action = np.argmax(Q[state])
        policy[state, best_action] = 1

    return V, policy


def value_iteration_w_ci(mdp: MDP, Q_tild, gamma: float = 0.95, theta: float = 0.01):
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

    return V, Q_tild, policy


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


if __name__ == "__main__":
    mdp = RiverSwimMDP()
    V, policy = value_iteration_q(mdp)
    V2, policy2 = value_iteration(mdp)

    print(V)
    print(policy)
    print("----------------")
    print(V2)
    print(policy2)
