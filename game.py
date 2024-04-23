# Imports
import numpy as np

from agents import QLearningAgent, SarsaAgent, DoubleQLearningAgent
from envs import GridA, GridB

# Constants
GRID_WORLD = "grid_a"  # 'grid_a' / 'grid_b'
AGENT = "qlearning"  # 'qlearning' / 'sarsa' / 'double_qlearning'

EPSILON = 0.1
EPSILON_VALUES = np.linspace(0.0, 1.0, num=6)  # 0, 0.2, 0.4, 0.6, 0.8, 1
ALPHA = 0.9
ALPHA_VALUES = np.linspace(0.0, 1.0, num=6)
START_STATE = (1, 1)
START_STATES = None  # to be completed

ACTIONS_NO = 4
STATES_NO = 70  # 7 * 10 grid

NO_EPISODES = 2000
MULTIAGENT = False


def get_agent():
    if not MULTIAGENT:
        if AGENT == "qlearning":
            return QLearningAgent(EPSILON, ALPHA, ACTIONS_NO, STATES_NO)
        elif AGENT == "sarsa":
            return SarsaAgent(EPSILON, ALPHA, ACTIONS_NO, STATES_NO)
        elif AGENT == "double_qlearning":
            return DoubleQLearningAgent(EPSILON, ALPHA, ACTIONS_NO, STATES_NO)
        else:
            raise ValueError("Not yet implemented agent!")
    else:
        # idk
        pass


def get_env():
    if not MULTIAGENT:
        if GRID_WORLD == "grid_a":
            return GridA(START_STATE)
        elif GRID_WORLD == "grid_b":
            return GridB(START_STATE)
    else:
        if GRID_WORLD == "grid_a":
            return GridA(START_STATES)
        elif GRID_WORLD == "grid_b":
            return GridB(START_STATES)


def main_play_single_agent():
    agent = get_agent()
    env = get_env()

    for episode in range(NO_EPISODES):
        state = env.reset()
        agent.new_episode()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update_internals(next_state, reward)

    agent.log_results()


def main_play_multi_agent():
    pass


if __name__ == "__main__":
    if MULTIAGENT:
        main_play_multi_agent()
    else:
        main_play_single_agent()
