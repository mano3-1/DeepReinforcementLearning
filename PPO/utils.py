import torch as T
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset

class PPO_memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.actions = []
        self.rewards = []
        self.states_ = []
        self.dones = []
    def remember(self, s, a, s_, r, done, p):
        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(p)
        self.dones.append(done)
        self.states_.append(s_)
        self.rewards.append(r)
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.dones.clear()
        self.states_.clear()
        self.rewards.clear()

class PPO_dataset(Dataset):
    def __init__(self,memory, device ):
        self.states = memory.states
        self.actions = memory.actions
        self.states_ = memory.states_
        self.rewards = memory.rewards
        self.dones = memory.dones
        self.probs = memory.probs
        self.device = device
    def __len__(self,):
        return len(self.states)
    def __getitem__(self, idx):
        s = T.tensor(self.states[idx]).to(self.device)
        s_  = T.tensor(self.states_[idx]).to(self.device)
        r = T.tensor(self.rewards[idx]).to(self.device)
        p = T.tensor(self.probs[idx]).to(self.device)
        a = T.tensor(self.actions[idx]).to(self.device)
        done = T.tensor(self.dones[idx]).to(self.device)
        return (s, a, s_, r, done, p)


class dense_critic(nn.Module):
    
    def __init__(self, lr, input_dims, fc1 = 128, fc2 = 256):
        super(dense_critic, self).__init__()
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
    
class dense_actor(nn.Module):
    
    def __init__(self, lr, input_dims, n_actions=2, fc1 = 128, fc2 = 256):
        super(dense_actor, self).__init__()
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

