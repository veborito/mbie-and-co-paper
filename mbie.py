import numpy as np
import gymnasium as gym
from mdps import MDP, RiverSwimMDP
from value_iteration import value_iteration


def confidence_transition(count: int, lambd: float, prob_vec_len: int):
    # pas sur de ça mais division par 0
    if count == 0:
        return 0
    return np.sqrt(2 * (np.log(2**prob_vec_len - 2) - np.log(lambd)) / count)


def confidence_reward(count: int, lambd: float):
    # pas très sur que ce soit correct
    # surtout cette partie
    if count == 0:
        return 0
    return np.sqrt(np.log(2 / lambd) / 2 * count)


class MBIE:
    def __init__(
        self,
        env: MDP or gym.env,
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
        self.count_sas = np.zeros(self.n_states, self.n_actions, self.n_states)
        self.count_sa = np.zeros(self.n_states, self.n_actions)
        self.Q_tild = np.zeros(self.n_states, self.n_actions)
        self.R_tild = np.zeros(self.n_states, self.n_actions)
        self.T_tild = np.zeros(self.n_states, self.n_actions, self.n_states)
        self.R_sum = np.zeros(self.n_states, self.n_actions)

    def _update_estimates(self, state, action):
        ci_reward = self.A * (self.max_reward / np.sqrt(self.count_sa[state, action]))
        self.R_tild[state, action] = (
            self.R_sum[state, action] / self.count_sa[state, action] + ci_reward
        )
        ci_trans = self.B * (1 / np.sqrt(self.count_sa[state, action]))
        self.T_tild[state, action] = (
            self.count_sas[state, action] / self.count_sa[state, action] + ci_trans
        )

    def build_estimates(self):
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.count_sa[s, a] == 0:
                    continue
                else:
                    self._update_estimates(s, a)
        return self.R_tild, self.T_tild

    def run(self):
        state = self.env.reset()


if __name__ == "__main__":
    mdp = RiverSwimMDP()
    V, policy = value_iteration(mdp, 0.95)  # simple value iteration yields 211_554
    print(V.sum())
    print(policy)
