import numpy as np
from mdps import MDP, RiverSwimMDP

# --------------------------- Q VALUE ITERATION FOR MBIE (NON OPTIMIZED) ------------------------------------

def q_one_step_lookahead_w_ci(mdp,Q_tild, R_tild, T_hat, T_ci, R_max, count_sa, state, gamma):
  V = np.max(Q_tild, axis=1)
  for action in range(mdp.n_actions):
      q_sum = 0
      if count_sa[state, action] == 0:
        q_sum = R_max / (1 - gamma)
      else:
        T_tild = np.copy(T_hat[state, action])
        epsilon = T_ci[state, action]
        best_next_state = np.argmax(V)
        # Add optimistic mass (epsilon / 2), capped so no probability exceeds 1.0
        delta = min(epsilon / 2, 1.0 - T_tild[best_next_state])
        T_tild[best_next_state] += delta
        
        to_remove = delta
        sorted_states = np.argsort(V)
        
        for next_state in sorted_states:
          if to_remove <= 0:
            break
          if next_state == best_next_state:
              continue
          # Take as much as possible without dropping probability below 0
          remove_amount = min(to_remove, T_tild[next_state])
          T_tild[next_state] -= remove_amount
          to_remove -= remove_amount
        reward = R_tild[state, action]
        q_sum = reward + gamma * np.sum(T_tild * V)
      Q_tild[state, action] = q_sum
  return Q_tild


def q_value_iteration_w_ci(
    mdp, Q_tild, R_tild, T_hat, T_ci, R_max,count_sa, gamma: float = 0.95, theta: float = 0.01
):
    V = np.zeros([mdp.n_states])
    delta = 1
    while theta < delta:
        delta = 0
        for state in range(mdp.n_states):
            Q_tild = q_one_step_lookahead_w_ci(
                mdp, Q_tild, R_tild, T_hat, T_ci, R_max,count_sa, state, gamma
            )
            update = np.max(Q_tild[state])

            delta = max(delta, np.abs(update - V[state]))
            V[state] = update

    policy = np.zeros([mdp.n_states, mdp.n_actions])
    for state in range(mdp.n_states):
        best_action = np.argmax(Q_tild[state])
        policy[state, best_action] = 1

    return V, Q_tild, policy

# --------------------------- BASIC Q VALUE ITERATION (NON OPTIMIZED) ------------------------------------

def q_one_step_lookahead(mdp, Q, state, gamma):
    for action in range(mdp.n_actions):
        q_sum = 0
        for next_state in range(mdp.n_states):
            prob = mdp.get_transition_probability(state, action, next_state)
            reward = mdp.get_reward(state, action)
            q_sum += prob * (reward + gamma * np.max(Q[next_state]))
        Q[state, action] = q_sum
    return Q


def q_value_iteration(mdp: MDP, gamma: float = 0.95, theta: float = 0.01):
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
        _Q = one_step_lookahead(mdp, V, state, gamma)
        best_action = np.argmax(_Q[state])
        policy[state, best_action] = 1

    return V, Q, policy

# --------------------------- BASIC VALUE ITERATION (NON OPTIMIZED) ------------------------------------

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


if __name__ == "__main__":
    mdp = RiverSwimMDP()
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    V2, policy2 = value_iteration(mdp)
