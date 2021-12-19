import collections
from collections import deque
import torch, gym, numpy as np, random

def change_to_tensor_float(np):
    return torch.from_numpy(np).float()
def change_to_tensor_integer(np):
    return torch.from_numpy(np).int()
def list_to_numpy(lst):
    return np.array(lst)

class Experience_Replay:
    def __init__(self):
        self.maxlen = 500
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.batch_size = 32
        self.min_len = 32

    def put(self, transition):  # transition: [s, a, r, s']
        self.buffer.append(transition)
    
    def sample(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []

        for transitions in mini_batch:
            state, action, reward, new_state, done = transitions
            s_list.append(state)
            a_list.append([action])
            r_list.append([reward])
            s_prime_list.append(new_state)
            done_list.append([done])
        
        return torch.tensor(list_to_numpy(s_list), dtype=torch.float), torch.tensor(list_to_numpy(a_list), dtype=torch.int64), \
            torch.tensor(list_to_numpy(r_list)), torch.tensor(list_to_numpy(s_prime_list), dtype=torch.float), \
                torch.tensor(list_to_numpy(done_list))
