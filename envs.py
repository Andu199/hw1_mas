import numpy as np
from enum import Enum


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

    def __init__(self, start_state):
        self.board = np.zeros((BaseEnv.HEIGHT, BaseEnv.WIDTH))
        self.board[3, 7] = CELL_TYPE.GOAL.value

        self.start_state = start_state
        self.act_state = start_state

    def step(self, action):
        pass

    def reset(self):
        self.board = np.zeros((BaseEnv.HEIGHT, BaseEnv.WIDTH))
        self.board[3, 7] = CELL_TYPE.GOAL.value

        self.act_state = self.start_state
        return BaseEnv.state_tuple_to_int(self.start_state)

    def print_board(self):
        for line in self.board:
            for val in line:
                print(val, end=' ')
            print()


class GridA(BaseEnv):
    def __init__(self, start_state):
        super().__init__(start_state)
        self.board[1: 5, 5] = CELL_TYPE.OBSTACLE.value

    def reset(self):
        super().reset()
        self.board[1: 5, 5] = CELL_TYPE.OBSTACLE.value
        return BaseEnv.state_tuple_to_int(self.start_state)

    def step(self, action):
        new_y, new_x = ACTION.apply_action(self.act_state, action)

        valid = True
        if not (0 <= new_y < BaseEnv.HEIGHT):
            valid = False

        if not (0 <= new_x < BaseEnv.WIDTH):
            valid = False

        if valid and self.board[new_y, new_x] == CELL_TYPE.OBSTACLE:
            valid = False

        if valid:
            self.act_state = (new_y, new_x)

        reward = -1
        done = False
        if self.board[self.act_state[0], self.act_state[1]] == CELL_TYPE.GOAL.value:
            reward = 1
            done = True

        return BaseEnv.state_tuple_to_int(self.act_state), reward, done


class GridB(BaseEnv):
    def __init__(self, start_state):
        super().__init__(start_state)
        self.wind_power = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def step(self, action):
        new_y, new_x = ACTION.apply_action(self.act_state, action)

        valid = True
        if not (0 <= new_y < BaseEnv.HEIGHT):
            valid = False

        if not (0 <= new_x < BaseEnv.WIDTH):
            valid = False

        if valid:
            self.act_state = (new_y, new_x)

        act_y, act_x = self.act_state
        act_y = max(0, act_y - self.wind_power[act_x])
        self.act_state = (act_y, act_x)

        reward = -1
        done = False
        if self.board[self.act_state[0], self.act_state[1]] == CELL_TYPE.GOAL.value:
            reward = 1
            done = True

        return BaseEnv.state_tuple_to_int(self.act_state), reward, done


if __name__ == "__main__":
    base = BaseEnv((3, 1))
    base.print_board()
