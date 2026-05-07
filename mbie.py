import numpy as np
import gymnasium as gym
from mdps import MDP, RiverSwimMDP
from value_iteration import value_iteration


class MBIE:
    def __init__(
        self,
        env: MDP or gym.Env,
        max_reward: int or float,
        discount_factor: float,
        A: float,
        B: float,
    ):
        self.env = env
        self.discount_factor = discount_factor  # discount factor

        self.max_reward = max_reward
        self.max_value = 1 / (1 - self.discount_factor)
        self.A = A
        self.B = B

        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.count_sas = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.count_sa = np.zeros([self.n_states, self.n_actions])
        self.Q_tild = np.full([self.n_states, self.n_actions], self.max_value)
        self.R_hat = np.zeros([self.n_states, self.n_actions])
        self.R_tild = np.zeros([self.n_states, self.n_actions])
        self.T_hat = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.T_tild = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.R_sum = np.zeros([self.n_states, self.n_actions])

    def _update_reward(self, s, a):
        reward_ci = self.A * (self.max_reward / np.sqrt(self.count_sa[s, a]))
        reward = self.R_sum[s, a] / self.count_sa[s, a]
        self.R_hat = reward
        self.R_tild[s, a] = reward + reward_ci

    def _update_transition(self, s, a, next_s):
        trans_ci = self.B * (1 / np.sqrt(self.count_sa[s, a]))
        prob = self.count_sas[s, a, next_s] / self.count_sa[s, a]
        self.T_hat[s, a, next_s] = prob
        self.T_tild[s, a, next_s] = prob + trans_ci

    def _normalize_transition(self, s, a):
        self.T_tild[s, a] /= self.T_tild[s, a].sum()

    def build_estimates(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.count_sa[s, a] == 0:
                    continue
                else:
                    self._update_reward(s, a)
                    for next_s in range(self.n_states):
                        self._update_transition(s, a, next_s)
                    # using normalized transitions may be incorrect
                    self._normalize_transition(s, a)

    def run(self, experiments):
        action = np.random.choice(self.n_actions)
        state = self.env.reset()
        for _ in range(experiments):
            next_state, reward, _, _ = self.env.step(action)
            self.count_sa[state, action] += 1
            self.count_sas[state, action, next_state] += 1
            self.R_sum[state, action] += reward
            self.build_estimates()


if __name__ == "__main__":
    mdp = RiverSwimMDP()
    V, policy = value_iteration(mdp, 0.95)
    print(V.sum())
    print(policy)
