from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object):
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.memory = deque([], maxlen=int(capacity))

    def append(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def length(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)