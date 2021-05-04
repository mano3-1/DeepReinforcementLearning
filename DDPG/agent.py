import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import gym
import random

class q_network(nn.Module):
    def __init__(self, lr, input_dims, a_dims, fc1=64, fc2=128):
        super(q_network,self).__init__()
        self.dense_action = nn.Linear(*a_dims, fc1//2)
        self.dense_state = nn.Linear((fc1//2) + input_dims[-1], fc1)
        self.dense2 = nn.Linear(fc1, fc2)
        self.dense3 = nn.Linear(fc2, 1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.to(self.device)
    def forward(self,state,action):
        state = state.to(self.device).float()
        action = action.to(self.device).float()
        action_en = F.relu(self.dense_action(action))
        input = T.cat([state,action_en], axis = -1)
        X = F.relu(self.dense_state(input))
        X = F.relu(self.dense2(X))
        X = self.dense3(X)
        return X
class actor(nn.Module):
    def __init__(self, lr, input_dims, a_dims, a, fc1=64, fc2 = 128):
        super(actor,self).__init__()
        self.dense1 = nn.Linear(*input_dims, fc1)
        self.dense2 = nn.Linear(fc1, fc2)
        self.dense3 = nn.Linear(fc2, *a_dims)
        self.a = a

        self.optimizer = optim.SGD(self.parameters(), lr = lr)
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.to(self.device)
    def forward(self, state):
        state = state.to(self.device).float()
        X = F.relu(self.dense1(state))
        X = F.relu(self.dense2(X))
        X = F.tanh(self.dense3(X))*self.a
        return X
        
class DDPG:
    def __init__(self, alpha, beta, gamma, input_dims, action_dims,
                 a, fc1=128, fc2=256, batch_size = 64):
        self.actorP = actor(alpha, input_dims, action_dims, a, fc1, fc2)
        self.actorT = actor(alpha, input_dims, action_dims, a, fc1, fc2)
        self.criticP = q_network(beta, input_dims, action_dims, fc1, fc2)
        self.criticT = q_network(beta, input_dims, action_dims, fc1, fc2)

        self.var = 0.1
        self.cntr = 0
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.max_mem = 100000
        self.tau = 0.1
        self.create_memory(max_mem=self.max_mem)

    def create_memory(self, max_mem):
        self.states = np.zeros((max_mem,*self.input_dims))
        self.actions = np.zeros((max_mem,*self.action_dims))
        self.dones = np.zeros((max_mem,1))
        self.next_states = np.zeros((max_mem,*self.input_dims))
        self.rewards = np.zeros((max_mem,1))
    def choose_action(self,state):
        self.cntr+=1
        state = T.tensor(state[np.newaxis]).to(self.actorP.device)
        action = self.actorP(state).cpu().detach().numpy()
        action = action + np.random.normal(0,self.var)
        self.var = np.clip(self.var*np.exp(-0.001*self.cntr), -float("inf"), 0.001)
        #print(action)
        return action
    def store(self, state, action, next_state, reward, done):
        index = self.cntr % self.max_mem
        self.states[index] = state
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done
        self.actions[index] = action
    def learn(self):
        if self.cntr<self.batch_size:
            return
        if self.cntr<self.max_mem:
            ids = random.choices([i for i in range(self.cntr)], k = self.batch_size)
        else:
            ids = random.choices([i for i in range(self.states.shape[0])], k=self.batch_size)
        state_batch, next_state_batch, reward_batch = self.states[ids], self.next_states[ids], self.rewards[ids]
        done_batch, action_batch = self.dones[ids], self.actions[ids]
        state_batch = T.tensor(state_batch).to(self.criticP.device)
        next_state_batch = T.tensor(next_state_batch).to(self.criticP.device)
        done_batch = T.tensor(done_batch).to(self.criticP.device)
        reward_batch = T.tensor(reward_batch).to(self.criticP.device)
        action_batch = T.tensor(action_batch).to(self.criticP.device)

        self.criticP.optimizer.zero_grad()
        qeval = self.criticP(state_batch, action_batch)
        qnext = self.criticT(next_state_batch, self.actorT(next_state_batch).detach()).detach()*(1-done_batch)
        qlabel = reward_batch + self.gamma*qnext
        qloss = ((qeval-qlabel)**2).sum(-1).to(self.actorP.device)
        qloss.mean().backward()
        self.criticP.optimizer.step()

        self.actorP.optimizer.zero_grad()
        policy_loss = -self.criticP(state_batch, self.actorP(state_batch)).to(self.actorP.device)
        policy_loss = policy_loss.sum(-1)
        policy_loss.mean().backward()
        self.actorP.optimizer.step()
    
    def update_target_nets(self):
        for target_param,local_param in zip(self.criticT.parameters(),self.criticP.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)
            
        for ptarget_param,plocal_param in zip(self.actorT.parameters(),self.actorP.parameters()):
            ptarget_param.data.copy_(self.tau*plocal_param.data+(1-self.tau)*ptarget_param.data)
    def save_weights(self):
        T.save(self.actorP.state_dict(),'/content/gdrive/My Drive/Reinforcement_Learning/ddpg/actor_best_1.pt')
        T.save(self.criticP.state_dict(),"/content/gdrive/My Drive/Reinforcement_Learning/ddpg/critic_best_1.pt")