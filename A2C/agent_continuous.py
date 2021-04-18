import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

class Actor(nn.Module):
    def __init__(self, lr, input_dims, n_actions, a_bound = 2):
        super(Actor, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.a_bound = a_bound
        
        #layers
        self.dense1 = nn.Linear(*self.input_dims, 64)
        self.dense2 = nn.Linear(64,128)
        self.mu_layer = nn.Linear(128,n_actions)
        self.sigma_layer = nn.Linear(128,n_actions)
        
        #devices and optimizer
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.optimizer = optim.SGD(self.parameters(), lr = self.lr)
        
        self.to(self.device)
    
    def forward(self, X):
        X = X.to(self.device).float()
        X = Mish(self.dense1(X))
        X = Mish(self.dense2(X))
        mu = self.a_bound*F.tanh(self.mu_layer(X))
        sigma = F.tanh(F.softplus(self.sigma_layer(X)))*0.1 + 1e-5
        return T.distributions.Normal(mu, sigma)
    
class Critic(nn.Module):
    def __init__(self, lr, input_dims):
        super(Critic, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        
        #layers
        self.dense1 = nn.Linear(*self.input_dims, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128,1)
        
        #optimizer and device
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.optimizer = optim.SGD(self.parameters(), lr = self.lr)
        self.to(self.device)
        
    def forward(self, X):
        X = X.to(self.device).float()
        X = Mish(self.dense1(X))
        X = Mish(self.dense2(X))
        X = self.dense3(X)
        return X

    
class A2C:
    def __init__(self, gamma, alpha, beta, input_dims, n_actions, a_bound = 2):
        self.gamma = gamma
        self.n_actions = n_actions
        
        self.actor = Actor(alpha, input_dims, n_actions, a_bound)
        self.critic = Critic(beta, input_dims)
        
    def make_tensor(self,X):
        X = np.asarray(X)
        X = [X]
        X = np.asarray(X)
        X = T.tensor(X, dtype=T.float)
        X = X.to(self.actor.device)
        return X
    
    def choose_action(self, state):
        state = self.make_tensor(state)
        #print(state.size())
        dists = self.actor(state)
        self.actions = dists.sample().cpu().detach().data.numpy()
        return self.actions
    def clip_grad_norm_(self, module, max_grad_norm):
        nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)
    def learn(self, state, reward, state_, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        state = self.make_tensor(state)
        reward = self.make_tensor(reward)
        state_ = self.make_tensor(state_)
        
        state_value = self.critic(state)
        state_value_ = self.critic(state_)
        
        advantage = reward + self.gamma*state_value_*(1-int(done)) - state_value
        critic_loss = (advantage**2).sum(-1)
        critic_loss.mean().backward()
        self.clip_grad_norm_(self.critic.optimizer, 0.5)
        self.critic.optimizer.step()
        
        norm_dists = self.actor(state)
        log_probs = norm_dists.log_prob(self.make_tensor(self.actions))
        actor_loss = -(log_probs*advantage.detach()).sum(-1) #don't forget to detach the advantage tensor
        actor_loss.mean().backward()
        self.clip_grad_norm_(self.actor.optimizer, 0.5)
        self.actor.optimizer.step()
        
    def save(self, PATH):
        T.save(self.actor.state_dict(), PATH + '/actor_best.pt')
        T.save(self.critic.state_dict(), PATH+'/critic_best.pt')
        
        
        
        