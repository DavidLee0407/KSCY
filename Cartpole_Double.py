import torch, gym, numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch.functional as F
import torch as nn

env = gym.make('CartPole-v0')
scores = []
losses = []
learning_rate, discount, epochs = 1e-3, 0.9, 1000
EPSILON_DECAY = 0.99
class DQNmodel(torch.nn.Module):
    def __init__(self):
        super(DQNmodel, self).__init__()

        self.feature = torch.nn.Sequential(
            torch.nn.Linear(4, 150),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 24),
            torch.nn.LeakyReLU())

        self.advantage =  torch.nn.Sequential(
            torch.nn.Linear(24, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 2))
    
        self.value =  torch.nn.Sequential(
            torch.nn.Linear(24, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1))

    def forward(self, state):
        state = self.feature(state)
        advantage, value = self.advantage(state), self.value(state)
        return (value + advantage - torch.mean(advantage))

loss_fn = torch.nn.MSELoss()
model = DQNmodel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def change_to_tensor(np):
    return torch.from_numpy(np).float()

def test_env(n):
    score_list = []
    env = gym.make('CartPole-v0')
    for i in range(n):
        done = False
        score = 0
        env.reset()
        init_state = env.reset()
        while not done:
            #env.render()
            qval = model(change_to_tensor(init_state))
            qval_ = qval.data.numpy()
            action = np.argmax(qval_)
            new_observation, reward, done, _ = env.step(action)
            init_state = new_observation
            score += reward
        score_list.append(score)
        env.close()
    return score_list

def train_model():
    env = gym.make('CartPole-v0')
    epsilon = 1
    run = 1

    for i in range(0, epochs):
        initial_state = env.reset()
        done = False
        score = 0

        while not done:
            qval = model(change_to_tensor(initial_state))
            if np.random.random() > epsilon:
                qval_ = qval.data.numpy()
                action = np.argmax(qval_)
            else:
                action = env.action_space.sample()
            
            new_state, reward, done, _ = env.step(action)
            initial_state = new_state
            score += reward
            if done: break
            
            with torch.no_grad():
                Q2 = model(change_to_tensor(new_state))
            max_Q2 = Q2.max().unsqueeze(0)

            Y = (reward + discount * max_Q2 * (1-done)).float()
            X = qval.squeeze()[action]

            loss = loss_fn(X, Y.detach())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        run += 1
        print(f'model 1: currently {round(run/epochs*100, 2)}% done')
        scores.append(score)
        if epsilon > 0.1:
            epsilon = epsilon * EPSILON_DECAY
        else:
            epsilon = 0

def draw_graph():
    plt.figure(figsize=(10,7))
    plt.plot(scores)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Rewards",fontsize=22)
    plt.show()

    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Losses",fontsize=22)
    plt.show()
    #torch.save(model, 'model_Test.pt')
    
train_model()
draw_graph()
print(test_env(3,))
