# Imports
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate

from agents import QLearningAgent, SarsaAgent, DoubleQLearningAgent
from envs import GridA, GridB

# Constants
GRID_WORLD = "grid_b"  # 'grid_a' / 'grid_b'
AGENT = "sarsa"  # 'qlearning' / 'sarsa' / 'double_qlearning'

EPSILON = 0.1
EPSILON_VALUES = [0.1, 0.5, 0.8]
ALPHA = 0.9
ALPHA_VALUES = [0.3, 0.6, 0.9]
START_STATE = (3, 0)
START_STATE_VALUES = [(0, 2), (3, 0), (6, 2)]
START_STATES = None  # to be completed

ACTIONS_NO = 4
STATES_NO = 70  # 7 * 10 grid

NO_EPISODES = 2000
TEST_EPISODES = 50
MULTIAGENT = False


def get_agent(epsilon, alpha, actions_no=ACTIONS_NO):
    if not MULTIAGENT:
        if AGENT == "qlearning":
            return QLearningAgent(epsilon, alpha, actions_no, STATES_NO)
        elif AGENT == "sarsa":
            return SarsaAgent(epsilon, alpha, actions_no, STATES_NO)
        elif AGENT == "double_qlearning":
            return DoubleQLearningAgent(epsilon, alpha, actions_no, STATES_NO)
        else:
            raise ValueError("Not yet implemented agent!")
    else:
        if AGENT == "qlearning":
            return QLearningAgent(epsilon, alpha, actions_no, STATES_NO)
        elif AGENT == "sarsa":
            return SarsaAgent(epsilon, alpha, actions_no, STATES_NO)
        elif AGENT == "double_qlearning":
            return DoubleQLearningAgent(epsilon, alpha, actions_no, STATES_NO)
        else:
            raise ValueError("Not yet implemented agent!")


def get_env(start_state):
    if not MULTIAGENT:
        if GRID_WORLD == "grid_a":
            return GridA(1, [start_state])
        elif GRID_WORLD == "grid_b":
            return GridB(1, [start_state])
    else:
        if GRID_WORLD == "grid_a":
            return GridA(3, [(0, 2), (3, 0), (6, 2)])


def print_q(q):
    WIDTH = 10
    HEIGHT = 7

    v_matrix = np.zeros((HEIGHT, WIDTH))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            v_matrix[y, x] = np.max(q[y * WIDTH + x])

    df = pd.DataFrame(v_matrix)
    return tabulate(df)


def log_results(results, key_name, task_id):
    plt.figure(figsize=(16, 9))
    agent_name = AGENT.replace("_", " ").capitalize()
    grid_name = GRID_WORLD.replace("_", " ").capitalize()

    os.makedirs(f"outputs/{task_id}/{GRID_WORLD}/{AGENT}", exist_ok=True)

    for key, value in results.items():
        rewards = value["rewards"]
        plt.plot(list(range(NO_EPISODES)), rewards, label=f"{key_name}: {key}")
    plt.title(f"Reward plots for {agent_name} {grid_name}; {key_name}")
    plt.legend()
    plt.savefig(f"outputs/{task_id}/{GRID_WORLD}/{AGENT}/{AGENT}_{GRID_WORLD}_rewards_{key_name}_task1.png")
    plt.clf()

    for key, value in results.items():
        steps = value["steps"]
        plt.plot(list(range(NO_EPISODES)), steps, label=f"{key_name}: {key}")
    plt.title(f"Steps per episode plots for {agent_name} {grid_name}; {key_name}")
    plt.legend()
    plt.savefig(f"outputs/{task_id}/{GRID_WORLD}/{AGENT}/{AGENT}_{GRID_WORLD}_steps_{key_name}_task1.png")
    plt.clf()

    logged_results = {}
    for key, value in results.items():
        del value["rewards"]
        del value["steps"]

        logged_results[key] = value

    with open(f"outputs/{task_id}/{GRID_WORLD}/{AGENT}/{AGENT}_{GRID_WORLD}_{key_name}.txt", "w") as f:
        json.dump(logged_results, f, indent=6)


def get_policy(agent):
    if hasattr(agent, "q"):
        pi = np.argmax(agent.q, axis=1)
    else:
        pi = np.argmax(agent.q1 + agent.q2, axis=1)

    return pi


def run_test_episodes(pi, env, no_test_episodes):
    steps_total = []
    for _ in range(no_test_episodes):
        state = env.reset()[0]
        steps = 0
        done = False

        while not done and steps < 1000:
            action = pi[state]
            next_state, reward, done = env.step([action])

            steps += 1
            state = next_state

        steps_total.append(steps)

    return np.std(steps_total)


def main_single_agent_run(agent, env):
    total_rewards = []
    total_steps = []

    old_pi = np.zeros(STATES_NO)
    converged_episode = -1
    robustness = []

    for episode in range(NO_EPISODES):
        print("EPISODE", episode)

        total_reward = 0.
        steps = 0

        state = env.reset()[0]
        agent.new_episode()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step([action])

            total_reward += reward
            steps += 1

            agent.update_internals(next_state, reward)
            state = next_state

        total_rewards.append(total_reward)
        total_steps.append(steps)

        pi = get_policy(agent)
        if all(pi == old_pi) and converged_episode == -1:
            converged_episode = episode
        old_pi = pi

        if (episode + 1) % 500 == 0:
            robustness.append(run_test_episodes(pi, env, TEST_EPISODES))

    return total_rewards, total_steps, converged_episode, np.mean(robustness)


def main_multi_agent_run(agents, env):
    total_rewards = []
    total_steps = []
    agents_num = len(agents)

    for episode in range(NO_EPISODES):
        print("EPISODE", episode)

        total_reward = 0.
        steps = 0

        states = env.reset()

        for ag in agents:
            ag.new_episode()
        done = False

        while not done:
            actions = []
            for i in range(agents_num):
                action = agents[i].get_action(states[i])
                actions.append(action)

            next_states, reward, done = env.step(actions)

            total_reward += reward
            steps += 1

            for i in range(agents_num):
                agents[i].update_internals(next_states[i], reward)
            states = next_states

        total_rewards.append(total_reward)
        total_steps.append(steps)

    return total_rewards, total_steps


def task1():
    if MULTIAGENT:
        raise ValueError("ERROR!!!")

    results = {}
    for epsilon in EPSILON_VALUES:
        agent = get_agent(epsilon, ALPHA)
        env = get_env(START_STATE)
        total_rewards, total_steps, converged_episode, robustness = main_single_agent_run(agent, env)

        results[epsilon] = {
            "rewards": total_rewards,
            "steps": total_steps,
            "avg_steps": np.mean(total_steps),
            "conv_episode": converged_episode,
            "robustness": robustness,
            "robustness_train": np.std(total_steps)
        }

    log_results(results, "epsilon", "task1")

    results = {}
    for alpha in ALPHA_VALUES:
        agent = get_agent(EPSILON, alpha)
        env = get_env(START_STATE)
        total_rewards, total_steps, converged_episode, robustness = main_single_agent_run(agent, env)

        results[alpha] = {
            "rewards": total_rewards,
            "steps": total_steps,
            "avg_steps": np.mean(total_steps),
            "conv_episode": converged_episode,
            "robustness": robustness,
            "robustness_train": np.std(total_steps)
        }

    log_results(results, "alpha", "task1")

    results = {}
    for start_state in START_STATE_VALUES:
        agent = get_agent(EPSILON, ALPHA)
        env = get_env(start_state)
        total_rewards, total_steps, converged_episode, robustness = main_single_agent_run(agent, env)

        results[str(start_state)] = {
            "rewards": total_rewards,
            "steps": total_steps,
            "avg_steps": np.mean(total_steps),
            "conv_episode": converged_episode,
            "robustness": robustness,
            "robustness_train": np.std(total_steps)
        }

    log_results(results, "start_state", "task1")


def task2():
    if MULTIAGENT:
        raise ValueError("ERROR!!!")

    results = {}
    for actions_no in [4, 8]:
        agent = get_agent(EPSILON, 0.3, actions_no)
        env = get_env(START_STATE)
        total_rewards, total_steps, converged_episode, robustness = main_single_agent_run(agent, env)

        results[actions_no] = {
            "rewards": total_rewards,
            "steps": total_steps,
            "avg_steps": np.mean(total_steps),
            "conv_episode": converged_episode,
            "robustness": robustness,
            "robustness_train": np.std(total_steps)
        }

    log_results(results, "no_actions", "task2")


def task3():
    if not MULTIAGENT:
        raise ValueError("ERROR!!!")

    agent1 = get_agent(0.3, ALPHA)
    agent2 = get_agent(0.3, ALPHA)
    agent3 = get_agent(0.3, ALPHA)
    env = get_env(START_STATE)

    main_multi_agent_run([agent1, agent2, agent3], env)
    print(print_q(agent1.q))
    print(print_q(agent2.q))
    print(print_q(agent3.q))


if __name__ == "__main__":
    # task1()
    # task2()
    task3()
