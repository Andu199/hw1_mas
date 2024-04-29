import numpy as np
from enum import Enum
from copy import deepcopy


class CELL_TYPE(Enum):
    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2


class ACTION(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    NORTH_EAST = 4
    SOUTH_EAST = 5
    SOUTH_WEST = 6
    NORTH_WEST = 7

    @staticmethod
    def apply_action(state, action):
        act_y, act_x = state

        if action == ACTION.NORTH.value:
            act_y -= 1
        elif action == ACTION.EAST.value:
            act_x += 1
        elif action == ACTION.SOUTH.value:
            act_y += 1
        elif action == ACTION.WEST.value:
            act_x -= 1

        if action == ACTION.NORTH_EAST.value:
            act_y -= 1
            act_x += 1
        elif action == ACTION.SOUTH_EAST.value:
            act_y += 1
            act_x += 1
        elif action == ACTION.SOUTH_WEST.value:
            act_y += 1
            act_x -= 1
        elif action == ACTION.NORTH_WEST.value:
            act_y -= 1
            act_x -= 1

        return act_y, act_x


class BaseEnv:
    WIDTH = 10
    HEIGHT = 7

    @staticmethod
    def state_tuple_to_int(state):
        y, x = state
        return y * BaseEnv.WIDTH + x
    @staticmethod
    def state_int_to_tuple(state):
        y = state // BaseEnv.WIDTH
        x = state % BaseEnv.WIDTH

        return y, x

    def __init__(self, num_agents, start_states):
        self.board = np.zeros((BaseEnv.HEIGHT, BaseEnv.WIDTH))
        self.board[3, 7] = CELL_TYPE.GOAL.value

        self.num_agents = num_agents

        self.start_states = {index : elem for index, elem in enumerate(start_states)}
        self.act_states = {index : elem for index, elem in enumerate(start_states)}

    def step(self, action):
        pass

    def reset(self):
        self.board = np.zeros((BaseEnv.HEIGHT, BaseEnv.WIDTH))
        self.board[3, 7] = CELL_TYPE.GOAL.value

        self.act_states = deepcopy(self.start_states)
        return [BaseEnv.state_tuple_to_int(v) for k, v in self.start_states.items()]

    def print_board(self):
        for line in self.board:
            for val in line:
                print(val, end=' ')
            print()


class GridA(BaseEnv):
    def __init__(self, num_agents, start_states):
        super().__init__(num_agents, start_states)
        self.board[1: 5, 5] = CELL_TYPE.OBSTACLE.value

    def reset(self):
        ans = super().reset()
        self.board[1: 5, 5] = CELL_TYPE.OBSTACLE.value
        return ans

    def step(self, action):
        if self.num_agents == 1:
            new_y, new_x = ACTION.apply_action(self.act_states[0], action[0])

            valid = True
            if not (0 <= new_y < BaseEnv.HEIGHT):
                valid = False

            if not (0 <= new_x < BaseEnv.WIDTH):
                valid = False

            if valid and self.board[new_y, new_x] == CELL_TYPE.OBSTACLE.value:
                valid = False

            if valid:
                self.act_states[0] = (new_y, new_x)

            reward = -1
            done = False
            if self.board[self.act_states[0][0], self.act_states[0][1]] == CELL_TYPE.GOAL.value:
                reward = 1
                done = True

            return BaseEnv.state_tuple_to_int(self.act_states[0]), reward, done

        else:
            on_goal = []
            for ag_id in range(self.num_agents):
                new_y, new_x = ACTION.apply_action(self.act_states[ag_id], action[ag_id])

                valid = True
                if not (0 <= new_y < BaseEnv.HEIGHT):
                    valid = False

                if not (0 <= new_x < BaseEnv.WIDTH):
                    valid = False

                if valid and self.board[new_y, new_x] == CELL_TYPE.OBSTACLE.value:
                    valid = False

                if valid:
                    self.act_states[ag_id] = (new_y, new_x)

                if self.board[self.act_states[ag_id][0], self.act_states[ag_id][1]] == CELL_TYPE.GOAL.value:
                    on_goal.append(True)
                else:
                    on_goal.append(False)

            agent_states = [BaseEnv.state_tuple_to_int(self.act_states[i]) for i in range(self.num_agents)]

            done = False
            reward = -1
            if all(on_goal):
                done = True
                reward = 10
            elif any(on_goal):
                done = True
                reward = -0.5

            return agent_states, reward, done


class GridB(BaseEnv):
    def __init__(self, num_agents, start_states):
        super().__init__(num_agents, start_states)
        self.wind_power = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def step(self, action):
        if self.num_agents != 1:
            raise ValueError("Grid B works only for single agent!")

        new_y, new_x = ACTION.apply_action(self.act_states[0], action[0])

        valid = True
        if not (0 <= new_y < BaseEnv.HEIGHT):
            valid = False

        if not (0 <= new_x < BaseEnv.WIDTH):
            valid = False

        if valid:
            self.act_states[0] = (new_y, new_x)

        act_y, act_x = self.act_states[0]
        act_y = max(0, act_y - self.wind_power[act_x])
        self.act_states[0] = (act_y, act_x)

        reward = -1
        done = False
        if self.board[self.act_states[0][0], self.act_states[0][1]] == CELL_TYPE.GOAL.value:
            reward = 1
            done = True

        return BaseEnv.state_tuple_to_int(self.act_states[0]), reward, done


if __name__ == "__main__":
    base = BaseEnv(1, [(1, 1)])
    base.print_board()
