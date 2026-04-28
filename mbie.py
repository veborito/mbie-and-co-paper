import numpy as np

def value_iteration(mdp, n_iterations, gamma, V = None):
    policy = np.zeros([mdp.n_states])
    assert(gamma > 0)
    assert(gamma < 1)
    # expected utility of states
    if (V is None):
        V = np.zeros([mdp.n_states])

    # expected utility of state-action pairs
    Q = np.zeros([mdp.n_states, mdp.n_actions])
    ## to fill in
    for n in range(n_iterations):
        a = policy(s)
        # rewards
        s_next = mdp.generate_next_state(s, a)
        Q[s,a] += 0.1 * (mdp.get_reward(s, a) + gamma * max(Q[s_next,:] - Q[s,a]))
        s = s_next
    return policy, V, Q


def estimated_rewards(R_estim, n_visited, state, action):
  count = n_visited[state, action]
  reward = R_estim[state, action] 

  return reward

def estimated_transitions(n_to_next_state, n_visited, state, action, next_state):
  pass

def mbie(mdp, m,gamma):
  n_visited = np.zeros(mdp.n_states, mdp.n_actions)
  n_to_next_state = np.zeros(mdp.n_states, mdp.n_actions, mdp.n_states)
  Q_estim = np.full((mdp.n_states, mdp.n_actions), 1 / (1 - gamma))
  R_estim = mdp.R.copy()
  T_estim = mdp.P.copy()
  
  for episode in range(1000):
    for _ in range(m)
    
  
