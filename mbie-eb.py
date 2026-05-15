import numpy as np
import gymnasium as gym
from mdps import MDP, RiverSwimMDP, SixArmsMDP
from mbie import MBIE
from value_iteration import q_value_iteration
from tqdm import tqdm
import matplotlib.pyplot as plt

class MBIE_EB(MBIE):
    def __init__(
        self,
        env: MDP,
        max_reward: float,
        discount_factor: float,
        C: float,
    ):
      super().__init__(env, max_reward, discount_factor, 0, 0)
      
      self.C = C
    
    def _update_reward(self, s, a):
        reward = self.R_sum[s, a] / self.count_sa[s, a] + self.C * self.max_reward / np.sqrt(self.count_sa[s, a])
        self.R_hat[s, a] = reward
    
    def _build_estimates(self, state, action):
        if self.count_sa[state, action] == 0:
          pass
        else:
            self._update_reward(state, action)
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
            _, self.Q_tild = q_value_iteration(
                self.env,
                self.Q_tild,
                self.R_hat,
                self.T_hat,
                gamma=self.discount_factor,
            )
            action = np.argmax(self.Q_tild[next_state])
            state = next_state

def runs(env, MAX_REWARD, GAMMA, A , B, len):
  results = []
  for _ in tqdm(range(len)):
    alg =  MBIE(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, A=A, B=B)
    alg.run(5000)
    results.append(alg.cumulative_reward())
  return results

if __name__ == "__main__":
    # MAX_REWARD = 10_000
    # GAMMA = 0.95
    # SEED = 42
    # C = 0.4
    # env = RiverSwimMDP()
    # alg = MBIE_EB(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, C=C)
    # alg2 = MBIE_EB(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, C=C)
    # alg.run(5000)
    # alg2.run(5000)
    # print(alg.cumulative_reward())
    # print(alg2.cumulative_reward())
  
    MAX_REWARD = 6_000
    C = 0.8
    GAMMA = 0.95
    env = SixArmsMDP()
    alg = MBIE_EB(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, C=C)
    alg2 = MBIE_EB(env=env, max_reward=MAX_REWARD, discount_factor=GAMMA, C=C)
    alg.run(5000)
    alg2.run(5000)
    print(alg.cumulative_reward())
    print(alg2.cumulative_reward())
