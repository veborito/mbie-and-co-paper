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

       # state 0 = center
        # states 1..6 = arm states

        self.state = 0

        #
        # From center state:
        #
        # each action tries to go to one arm
        #

        success_probs = [
            0.1,   # arm 1 (best reward, hardest)
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]

        rewards = [
            50.0,
            20.0,
            10.0,
            5.0,
            2.0,
            1.0,
        ]

        for action in range(6):

            arm_state = action + 1
            p = success_probs[action]

            #
            # From center:
            #
            # with probability p:
            #   go to arm state
            #
            # otherwise:
            #   stay in center
            #

            self.P[0, action, arm_state] = p
            self.P[0, action, 0] = 1.0 - p

            #
            # reward only for successful transition
            #
            self.R[0, action] = rewards[action] * p

        #
        # From arm states:
        #
        # any action deterministically returns to center
        #

        for s in range(1, 7):
            for a in range(6):
                self.P[s, a, 0] = 1.0

    def reset(self):
        self.state = 0
        return self.state
