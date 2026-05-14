import numpy as np
import gymnasium as gym
from mdps import MDP, RiverSwimMDP, SixArmsMDP
from value_iteration import q_value_iteration_w_ci
from tqdm import tqdm

class MBIE:
    def __init__(
        self,
        env: MDP,
        max_reward: float,
        discount_factor: float,
        A: float,
        B: float,
    ):
        self.env = env
        self.discount_factor = discount_factor  # discount factor

        self.max_reward = max_reward
        self.max_value = max_reward / (1 - self.discount_factor)
        self.A = A
        self.B = B

        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.policy = np.zeros([self.n_states, self.n_actions])
        self.count_sas = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.count_sa = np.zeros([self.n_states, self.n_actions])
        self.Q_tild = np.full([self.n_states, self.n_actions], self.max_value)
        self.R_hat = np.zeros([self.n_states, self.n_actions])
        self.R_tild = np.zeros([self.n_states, self.n_actions])
        self.T_hat = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.T_ci = np.zeros([self.n_states, self.n_actions])
        self.R_sum = np.zeros([self.n_states, self.n_actions])

    def _update_reward(self, s, a):
        reward_ci = self.A * (self.max_reward / np.sqrt(self.count_sa[s, a]))
        reward = self.R_sum[s, a] / self.count_sa[s, a]
        self.R_hat[s, a] = reward
        self.R_tild[s, a] = reward + reward_ci

    def _update_transition(self, s, a, next_s):
        prob = self.count_sas[s, a, next_s] / self.count_sa[s, a]
        self.T_hat[s, a, next_s] = prob
    
    def _update_transition_ci(self, s, a):
        trans_ci = self.B * (1 / np.sqrt(self.count_sa[s, a]))
        self.T_ci[s, a] = trans_ci

    def cumulative_reward(self):
        return self.R_sum.sum()

    def _build_estimates(self, state, action):
        if self.count_sa[state, action] == 0:
          pass
        else:
            self._update_reward(state, action)
            self._update_transition_ci(state, action)
            for next_s in range(self.n_states):
                self._update_transition(state, action, next_s)
                
    def run(self, experiments):
        state = self.env.reset()
        action = np.argmax(self.Q_tild[state])
        for _ in tqdm(range(experiments)):
            next_state, reward, _, _ = self.env.step(action)
            self.count_sa[state, action] += 1
            self.count_sas[state, action, next_state] += 1
            self.R_sum[state, action] += reward
            self._build_estimates(state, action)
            _, self.Q_tild, _ = q_value_iteration_w_ci(
                self.env,
                self.Q_tild,
                self.R_tild,
                self.T_hat,
                self.T_ci,
                self.max_reward,
                self.count_sa,
                gamma=self.discount_factor,
            )
            action = np.argmax(self.Q_tild[next_state])
            state = next_state

def run(alg):
  alg.run(5000)
  print(alg.cumulative_reward())

if __name__ == "__main__":
    MAX_REWARD = 10_000
    GAMMA = 0.95
    SEED = 42
    A = 0.3
    B = 0.0
    # env = RiverSwimMDP()
    # alg = MBIE(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, A=A, B=B)
    # alg2 = MBIE(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, A=A, B=B)
    # alg.run(5000)
    # alg2.run(5000)
    # print(alg.cumulative_reward())
    # print(alg2.cumulative_reward())
    MAX_REWARD = 6_000
    A = 0.3
    B = 0.08
    env = SixArmsMDP()
    alg = MBIE(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, A=A, B=B)
    # alg2 = MBIE(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, A=A, B=B)
    alg.run(5000)
    # alg2.run(5000)
    print(alg.cumulative_reward())
    # print(alg2.cumulative_reward())
