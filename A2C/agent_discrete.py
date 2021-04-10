import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np

class critic(nn.Module):
    
    def __init__(self, lr, input_dims, fc1 = 128, fc2 = 256):
        super(critic, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1 = fc1
        self.fc2 = fc2
        
        #layers
        self.dense1 = nn.Linear(*self.input_dims, self.fc1)
        self.dense2 = nn.Linear(self.fc1, self.fc2)
        self.dense3 = nn.Linear(self.fc2, 1)
        
        #optimizers and devices
        self.optimizer = optim.SGD(self.parameters(), lr = self.lr)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.to(self.device)
        
    def forward(self, X):
        X = X.float().to(self.device)
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = self.dense3(X)
        return X
    
class actor(nn.Module):
    
    def __init__(self, lr, input_dims, n_actions=2, fc1 = 128, fc2 = 256):
        super(actor, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1 = fc1
        self.fc2 = fc2
        
        #layers
        self.dense1 = nn.Linear(*self.input_dims, self.fc1)
        self.dense2 = nn.Linear(self.fc1, self.fc2)
        self.dense3 = nn.Linear(self.fc2, n_actions)
        
        #optimizers and devices
        self.optimizer = optim.SGD(self.parameters(), lr = self.lr)
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        
        self.to(self.device)
        
    def forward(self, X):
        X = X.float().to(self.device)
        X = F.relu(self.dense1(X))
        X = F.relu(self.dense2(X))
        X = T.nn.Softmax(dim=-1)(self.dense3(X)) 
        return X
    
class A2C:
    
    def __init__(self, gamma, alpha, beta, input_dims, n_actions, fc1 = 64, fc2 = 128):     
        self.gamma = gamma
        self.alpha = alpha
        self.beta  = beta
        self.n_actions = n_actions
        self.input_dims = input_dims
        
        self.actor = actor(alpha, input_dims, n_actions, fc1, fc2)
        self.critic = critic(beta, input_dims, fc1, fc2)
        
    def make_tensor(self, state):
        state = np.asarray(state)
        state = [state]
        state = np.asarray(state)
        state = T.tensor(state)
        return state
    
    def choose_action(self, state):
        state = self.make_tensor(state)
        action_probs = self.actor(state)
        action_probs = action_probs.cpu().detach().numpy()
        action_probs = action_probs[0] #removing batch dims
        
        #sample actions from the action space. One can choose to follow greedy policy as well, 
        #but sampling can also allows the agent to explore.
        action = np.random.choice(np.arange(action_probs.shape[0]),p = action_probs.ravel()) 
        self.action_one_hots = np.zeros(self.n_actions)
        self.action_one_hots[action] = 1
        self.action_one_hots = self.make_tensor(self.action_one_hots)
        return action
    
    def learn(self, state, reward, state_, done):      
        #zero_grads
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        #state tensors
        state = self.make_tensor(state)
        state_ = self.make_tensor(state_)
        reward = self.make_tensor(reward)
        
        #values
        critic_value_ = self.critic(state_)
        critic_value = self.critic(state)
        
        #advantage
        advantage = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
        
        #critic loss
        critic_loss = (advantage**2).sum(-1)
        critic_loss.mean().backward()
        self.critic.optimizer.step()

        #actor loss
        log_probs = self.actor(state)
        #print(log_probs)
        log_probs = T.log(log_probs)
        
        # .detach() is necessary to detach the advantage tensor from critic
        actor_loss = -(self.action_one_hots*advantage.detach()*log_probs).sum(-1) 
        actor_loss.mean().backward()
        self.actor.optimizer.step()
        


        
        
        
        
        
        