import numpy as np
import gymnasium as gym


class MDP:
    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state = 0
        self.P = np.zeros([n_states, n_actions, n_states])
        self.R = np.zeros([n_states, n_actions])

    def get_transition_probability(
        self, state: int, action: int, next_state: int
    ) -> float:
        return self.P[state, action, next_state]

    def get_transition_probabilities(self, state: int, action: int) -> list[float]:
        return self.P[state, action]

    def generate_state(self, state: int, action: int) -> int:
        return np.random.choice(self.n_states, p=self.P[state, action])

    def get_reward(self, state: int, action: int) -> float:
        return self.R[state, action]

    def get_rewards(self, state: int) -> list[int]:
        return self.R[state]

    def step(self, action: np.intp) -> tuple[int, float, bool, dict]:
        # respect gymnasium step for extensions
        reward = self.get_reward(self.state, action)
        self.state = self.generate_state(self.state, action)
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state


class RiverSwimMDP(MDP):
    def __init__(self):
        super().__init__(6, 2)

        self.state = np.random.choice([1, 2])
        self.P = np.array(
            [  # action 0           #action 1
                [[1, 0, 0, 0, 0, 0], [0.7, 0.3, 0, 0, 0, 0]],  # state 0
                [[1, 0, 0, 0, 0, 0], [0.1, 0.6, 0.3, 0, 0, 0]],  # state 1
                [[0, 1, 0, 0, 0, 0], [0, 0.1, 0.6, 0.3, 0, 0]],  # state 2
                [[0, 0, 1, 0, 0, 0], [0, 0, 0.1, 0.6, 0.3, 0]],  # state 3
                [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0.1, 0.6, 0.3]],  # state 4
                [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0.7, 0.3]],  # state 5
            ]
        )

        self.R = np.array(
            [
                [5, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 3000],
            ]
        )

    def reset(self):
        self.state = np.random.choice([1, 2])
        return self.state


# AI generated
class SixArmsMDP(MDP):
    def __init__(self):
        super().__init__(7, 6)
        self.state = 0

        self.P = np.array(
          [
            # STATE 0 (center)
            [
              # action 0
              [0, 1, 0, 0, 0, 0, 0],
              # action 1
              [0.85, 0, 0.15, 0, 0, 0, 0],
              # action 2
              [0.9, 0, 0, 0.1, 0, 0, 0],
              # action 3
              [0.95, 0, 0, 0, 0.05, 0, 0],
              # action 4
              [0.97, 0, 0, 0, 0, 0.03, 0],
              # action 5
              [0.99, 0, 0, 0, 0, 0, 0.01],
            ],

            # STATE 1
            [
              # action 0
              [0, 1, 0, 0, 0, 0, 0],
              # action 1
              [0, 1, 0, 0, 0, 0, 0],
              # action 2
              [0, 1, 0, 0, 0, 0, 0],
              # action 3
              [0, 1, 0, 0, 0, 0, 0],
              # action 4
              [1, 0, 0, 0, 0, 0, 0],
              # action 5
              [0, 1, 0, 0, 0, 0, 0],
            ],
            #
            # STATE 2
            #
            [
              # action 0
              [1, 0, 0, 0, 0, 0, 0],
              # action 1
              [0, 0, 1, 0, 0, 0, 0],
              # action 2
              [1, 0, 0, 0, 0, 0, 0],
              # action 3
              [1, 0, 0, 0, 0, 0, 0],
              # action 4
              [1, 0, 0, 0, 0, 0, 0],
              # action 5
              [1, 0, 0, 0, 0, 0, 0],
            ],

            #
            # STATE 3
            #
            [
              # action 0
              [1, 0, 0, 0, 0, 0, 0],
              # action 1
              [1, 0, 0, 0, 0, 0, 0],
              # action 2
              [0, 0, 0, 1, 0, 0, 0],
              # action 3
              [1, 0, 0, 0, 0, 0, 0],
              # action 4
              [1, 0, 0, 0, 0, 0, 0],
              # action 5
              [1, 0, 0, 0, 0, 0, 0],
            ],

            #
            # STATE 4
            #
            [
                # action 0
              [1, 0, 0, 0, 0, 0, 0],
              # action 1
              [1, 0, 0, 0, 0, 0, 0],
              # action 2
              [1, 0, 0, 0, 0, 0, 0],
              # action 3
              [0, 0, 0, 0, 1, 0, 0],
              # action 4
              [1, 0, 0, 0, 0, 0, 0],
              # action 5
              [1, 0, 0, 0, 0, 0, 0],
            ],

            #
            # STATE 5
            #
            [
              # action 0
              [1, 0, 0, 0, 0, 0, 0],
              # action 1
              [1, 0, 0, 0, 0, 0, 0],
              # action 2
              [1, 0, 0, 0, 0, 0, 0],
              # action 3
              [1, 0, 0, 0, 0, 0, 0],
              # action 4
              [0, 0, 0, 0, 0, 1, 0],
              # action 5
              [1, 0, 0, 0, 0, 0, 0],
            ],

            #
            # STATE 6
            #
            [
                # action 0
              [1, 0, 0, 0, 0, 0, 0],
              # action 1
              [1, 0, 0, 0, 0, 0, 0],
              # action 2
              [1, 0, 0, 0, 0, 0, 0],
              # action 3
              [1, 0, 0, 0, 0, 0, 0],
              # action 4
              [1, 0, 0, 0, 0, 0, 0],
              # action 5
              [0, 0, 0, 0, 0, 0, 1],
            ],
          ]
        )

        #
        # expected rewards
        #
        # expected reward placed directly on
        # center-state actions
        #
        # E[r] = p * reward
        #
                
        room_reward = [
          0,
          50,
          133,
          300,
          800,
          1660,
          6000,
        ]
        
        for s, states in enumerate(self.P):
          for a, actions in enumerate(states):
            for n_s, prob in enumerate(actions):
              self.R[s, a] += prob * room_reward[n_s]

    def reset(self):
        self.state = 0
        return self.state

if __name__== "__main__":
  mdp = SixArmsMDP()
