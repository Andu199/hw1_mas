import random

import numpy as np


def epsilon_greedy(state, q, epsilon):
    states_no, actions_no = q.shape
    prob = random.random()

    if prob < epsilon:
        return random.randint(0, actions_no - 1)
    else:
        return q[state].argmax()


class BasicAgent:
    def __init__(self, epsilon, alpha, actions_no, states_no):
        self.epsilon = epsilon
        self.alpha = alpha
        self.actions_no = actions_no
        self.states_no = states_no

    def get_action(self, state):
        """
        Used to get the action to be executed based on the state that the agent is found in.
        :param state: agent's state
        :return: an action
        """
        pass

    def new_episode(self):
        """
        Used to inform the agent of a new episode.
        """
        pass

    def update_internals(self, next_state, reward):
        """
        Used to update the Q values after one step in the environment.
        :param next_state: agent's new state
        :param reward: agent's reward for the action it previously executed
        """
        pass

    def log_results(self):
        """
        Used to log the final results (a set of metrics).
        :return: a set of metrics
        """
        pass


class QLearningAgent(BasicAgent):
    def __init__(self, epsilon, alpha, actions_no, states_no):
        super().__init__(epsilon, alpha, actions_no, states_no)

        self.q = np.zeros((states_no, actions_no))
        self.current_state = None
        self.current_action = None

    def get_action(self, state):
        self.current_state = state
        self.current_action = epsilon_greedy(state, self.q, self.epsilon)
        return self.current_action

    def update_internals(self, next_state, reward):
        self.q[self.current_state, self.current_action] +=\
            self.alpha * (reward + self.q[next_state].max() - self.q[self.current_state, self.current_action])


class SarsaAgent(BasicAgent):
    def __init__(self, epsilon, alpha, actions_no, states_no):
        super().__init__(epsilon, alpha, actions_no, states_no)

        self.q = np.zeros((states_no, actions_no))
        self.current_state = None
        self.current_action = None

    def new_episode(self):
        self.current_action = None

    def get_action(self, state):
        self.current_state = state
        if self.current_action is None:
            self.current_action = epsilon_greedy(state, self.q, self.epsilon)
        return self.current_action

    def update_internals(self, next_state, reward):
        next_action = epsilon_greedy(next_state, self.q, self.epsilon)

        self.q[self.current_state, self.current_action] +=\
            self.alpha * (reward + self.q[next_state, next_action] - self.q[self.current_state, self.current_action])
        self.current_action = next_action


class DoubleQLearningAgent(BasicAgent):
    def __init__(self, epsilon, alpha, actions_no, states_no):
        super().__init__(epsilon, alpha, actions_no, states_no)

        self.q1 = np.array((states_no, actions_no))
        self.q2 = np.array((states_no, actions_no))

        self.current_state = None
        self.current_action = None

    def get_action(self, state):
        self.current_state = state
        self.current_action = epsilon_greedy(state, self.q1 + self.q2, self.epsilon)
        return self.current_action

    def update_internals(self, next_state, reward):
        prob = random.random()
        if prob < 0.5:
            best_a = self.q2[next_state].argmax()
            self.q1[self.current_state, self.current_action] += \
                self.alpha * (reward + self.q1[next_state, best_a] - self.q1[self.current_state, self.current_action])
        else:
            best_a = self.q1[next_state].argmax()
            self.q2[self.current_state, self.current_action] += \
                self.alpha * (reward + self.q2[next_state, best_a] - self.q2[self.current_state, self.current_action])
