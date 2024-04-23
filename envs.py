class BaseEnv:
    def __init__(self, start_state):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass


class GridA(BaseEnv):
    def __init__(self, start_state):
        super().__init__(start_state)

    def step(self, action):
        pass


class GridB(BaseEnv):
    def __init__(self, start_state):
        super().__init__(start_state)

    def step(self, action):
        pass
