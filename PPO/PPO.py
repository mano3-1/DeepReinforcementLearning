import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from torch.distributions import Categorical

import torch as T
from torch.nn import functional as F
 
from utils import dense_critic, dense_actor, PPO_memory, PPO_dataset

class Agent:
    def __init__(self, a_lr, c_lr, state_dims, n_actions,
                    c1 = 0.5, c2 = 0.001, print_loss =False, gamma = 0.99, eps_clip = 0.2, ckpt_dir = "./"):
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.gamma = gamma
        self.state_dims = state_dims
        self.eps_clip = eps_clip
        self.print_loss = print_loss
        self.n_actions = n_actions
        self.c1 = c1
        self.c2 = c2
        self.a_ckpt_dir = ckpt_dir + "actor.pth"
        self.c_ckpt_dir = ckpt_dir + "critic.pth"


        self.critic = dense_critic(c_lr, state_dims)
        self.actor = dense_actor(a_lr, state_dims, n_actions = n_actions)
        self.memory = PPO_memory()

    def save_weights(self,):
        print("saving models...")
        T.save(self.critic.state_dict(), self.c_ckpt_dir)
        T.save(self.actor.state_dict(), self.a_ckpt_dir)

    def load_models(self,):
        print("loading models...")
        self.critic.load_state_dict(T.load(self.c_ckpt_dir, map_location=self.critic.device))
        self.actor.load_state_dict(T.load(self.a_ckpt_dir, map_location=self.actor.device))

    def remember(self,s, a, s_, r, done, logp):
        self.memory.remember(s,a,s_,r,done,logp)
        
    def choose_action(self, state):
        state = T.tensor(state).to(self.actor.device)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().cpu().item(), action_logprob.detach().cpu().float().item()
    
    def estimate(self, state, actions):
        #compute new probs
        probs = self.actor(state)
        dist = Categorical(probs)
        newlogprob = dist.log_prob(actions)
        entropy = dist.entropy()

        #compute values
        values = self.critic(state)
        
        return newlogprob, values, entropy

    def learn(self, epochs):

        #monte carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            discounted_reward = reward + self.gamma*(1-done)*discounted_reward
            rewards.insert(0, discounted_reward)

        #normalize the rewards
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic.device)
        rewards = (rewards - rewards.mean())/(rewards.std() + 1e-7)
        
        #convert list to tensor
        old_states = T.tensor(self.memory.states).to(self.critic.device)
        old_actions = T.tensor(self.memory.actions).to(self.critic.device)
        old_logprobs = T.tensor(self.memory.logprobs).to(self.critic.device)
        
        
        for i in range(epochs):
            
            #computing ratio
            newlogprob , state_values, entropy = self.estimate(old_states, old_actions)
            ratios = T.exp(newlogprob-old_logprobs.detach())
            
            #computing advantage
            advantages = rewards - state_values.detach()
            
            #surrogate loss
            surr1 = ratios*advantages
            surr2 = T.clamp(ratios, 1-self.eps_clip,1+self.eps_clip)*advantages

            #total policy loss
            policy_loss = -T.min(surr1, surr2) - self.c2*entropy
            self.actor.optimizer.zero_grad()
            policy_loss.mean().backward()
            self.actor.optimizer.step()

            #critic_loss
            critic_loss = (rewards-state_values)**2
            self.critic.optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic.optimizer.step()

            if i%4==0 and self.print_loss:
                print("epoch : {} | policy_loss : {} | critic_loss : {}".format(i,
                                                             policy_loss.mean().item(),
                                                             critic_loss.mean().item()))
        
        self.memory.clear()