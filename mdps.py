import numpy as np


class MDP:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = np.zeros([n_states, n_actions, n_states])
        self.R = np.zeros([n_states, n_actions])

    def get_transition_probability(self, state, action, next_state):
        return self.P[state, action, next_state]

    def generate_state(self, state, action):
        return np.random.choice(self.n_states, p=self.P[state, action])

    def get_reward(self, state, action):
        return self.R[state, action]


class RiverSwimMDP(MDP):
    def __init__(self):
        super().__init__(6, 2)
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


class SixArmsMDP(MDP):
    def __init__(self):
        super().__init__(7, 6)


"""
TODO:
    Make this mess prettier with a class

"""


class Model:
    def __init__(self):
        pass


def confidence_transition(count: int, lambd: float, prob_vec_len: int):
    # pas sur de ça mais division par 0
    if count == 0:
        return 0
    return np.sqrt(2 * (np.log(2**prob_vec_len - 2) - np.log(lambd)) / count)


# pas très sur que ce soit correct
def confidence_reward(count: int, lambd: float):
    # surtout cette partie
    if count == 0:
        return 0
    return np.sqrt(np.log(2 / lambd) / 2 * count)


def value_iteration(mdp: MDP, gamma: float, lmbd_R: float, lmbd_T: float):
    count_sa = np.zeros(mdp.n_states, mdp.n_actions)
    count_sas = np.zeros(mdp.n_states, mdp.n_actions, mdp.n_states)
    Q_estim = np.full(mdp.P.shape, (1 / (1 - gamma)))
    T_estim = np.zeros_like(mdp.P)
    R_estim = np.zeros_like(mdp.R)

    for _ in range(5000):
        for state in range(mdp.n_states):
            pass


def update_Q(mdp, count_sa, Q_estim, R_estim, T_estim, gamma, lmbd_R, lmbd_T):
    conf_matrix_T = np.zeros_like(T_estim)
    conf_matrix_R = np.zeros_like(R_estim)
    # calculate all values + conf interval
    # logic is good I think
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            # calculate conf_matrix_R
            conf_matrix_R[s, a] = R_estim[s, a] + confidence_reward(
                count_sa[s, a], lmbd_R
            )

            for next_s in range(mdp.n_states):
                # calculate conf_matrix_T
                conf_matrix_T[s, a, next_s] = T_estim[
                    s, a, next_s
                ] + confidence_transition(count_sa[s, a], lmbd_T, mdp.n_states)

    # pseudocode for update
    for s in range(mdp.n_staes):
        for a in range(mdp.n_actions):
            pass


if __name__ == "__main__":
    mdp = RiverSwimMDP()
