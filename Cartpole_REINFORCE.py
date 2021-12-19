import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb

env = gym.make('CartPole-v0')
EPSILON_DECAY = 0.99
MAX_DUR = 200
MAX_EPISODES = 1000
RECORD_EVERY = 25
gamma = 0.9
duration = [] 
losses = []

model = torch.nn.Sequential(
    torch.nn.Linear(4, 150),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(150, 2),
    torch.nn.Softmax(dim=0) 
)

learning_rate = 9e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def discount_rewards(rewards, gamma=0.9):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards 
    disc_return /= disc_return.max() 
    return disc_return

def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))

def train_model():

    env = gym.make('CartPole-v0')
    run = 1

    for _ in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False 
        transitions = [] 
        
        for _ in range(MAX_DUR): 
            act_prob = model(torch.from_numpy(curr_state).float()) 
            action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) 
            prev_state = curr_state
            curr_state, reward, done, _ = env.step(action) 
            transitions.append((prev_state, action, reward)) 
            
            if done: 
                duration.append(len(transitions))
                break

        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) 
        disc_returns = discount_rewards(reward_batch) 
        state_batch = torch.Tensor([s for (s,a,r) in transitions]) 
        action_batch = torch.Tensor([a for (s,a,r) in transitions]) 
        pred_batch = model(state_batch)
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
        loss = loss_fn(prob_batch, disc_returns)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run += 1
        print(f'model 1: currently {round(run/MAX_EPISODES*100, 2)}% done')

def draw_graph():
    plt.figure(figsize=(10,7))
    plt.plot(duration)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Rewards",fontsize=22)
    plt.show()

    plt.plot(losses)
    plt.xlabel("Episodes",fontsize=22)
    plt.ylabel("Losses",fontsize=22)
    plt.show()
    #torch.save(model, 'model_Test.pt')
    
def test_env(n):
    score_list = []
    env = gym.make('CartPole-v0')
    for i in range(n):
        done = False
        score = 0
        env.reset()
        init_state = env.reset()
        while not done:
            act_prob = model(torch.from_numpy(init_state).float()) 
            action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) 
            new_state, reward, done, _ = env.step(action)
            init_state = new_state
            score += reward
        score_list.append(score)
        env.close()
    return score_list

train_model()
draw_graph()
print(test_env(3))

