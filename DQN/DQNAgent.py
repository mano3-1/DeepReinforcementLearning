import torch as T
from torch import nn
from torch import optim as optim
from torch.nn import functional as F

import numpy as np

class DQN(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DQN,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        c,_,_ = input_dims
        self.conv1 = nn.Conv2d(c,8,(4,4),stride=(2,2))
        self.conv2 = nn.Conv2d(8,16,(3,3),stride=(2,2))
        self.conv3 = nn.Conv2d(16,32,(2,2),stride=(2,2))
        self.conv4 = nn.Conv2d(32,64,(3,3),stride=(2,2))
        self.maxpooling = nn.MaxPool2d((3,3))
        self.flatten = nn.Flatten()
        self.Dense = nn.Linear(64,n_actions)
        
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.optimizer = optim.SGD(self.parameters(), lr = lr)
        self.to(self.device)
        self.loss = nn.SmoothL1Loss()
        
    def forward(self,X):
        X = X/255.0
        X = X.float()
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = self.maxpooling(X)
        X = self.flatten(X)
        X = F.relu(self.Dense(X))
        return X
    
class DQN_fc(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1 = 512, fc2 = 256):
        super(DQN_fc,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
       
        self.fc1 = fc1 
        self.fc2 = fc2
        
        self.fc1_l = nn.Linear(*self.input_dims,self.fc1)
        self.fc2_l = nn.Linear(self.fc1,self.fc2)
        self.final = nn.Linear(self.fc2,self.n_actions)
        
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.optimizer = optim.SGD(self.parameters(), lr = lr)
        self.to(self.device)
        self.loss = nn.MSELoss()
        
    def forward(self,X):
        X = X.float()
        X = F.relu(self.fc1_l(X))
        X = F.relu(self.fc2_l(X))
        X = F.relu(self.final(X))
        return X
    
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, tau = 0.89,
                 max_mem_size=1000, eps_end=0.001, eps_dec=5e-4,fc = True ,fc1 = None,fc2 = None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0
        self.tau = tau
    
        if fc:
            self.Q_eval = DQN_fc(lr,input_dims,n_actions,fc1,fc2)
            self.Q_target = DQN_fc(lr,input_dims,n_actions,fc1,fc2)
        else:
            self.Q_eval = DQN(lr, input_dims, n_actions)
            self.Q_target = DQN(lr, input_dims, n_actions)
        self.state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)
        
    def store_actions(self,state,action,state_,reward,done):
        index = self.mem_cntr%self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr+=1
        
    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_cntr<self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=True)
        
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval(state_batch)[batch_index,action_batch]
        q_next = self.Q_target(new_state_batch).max(1)[0].detach()
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma*q_next
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon-self.eps_dec if self.epsilon>self.eps_min else self.eps_min
    
    def train_targetnet(self):
        for target_param,local_param in zip(self.Q_target.parameters(),self.Q_eval.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)
        
        