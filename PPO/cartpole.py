import gym
from PPO import Agent
import numpy as np

env = gym.make("CartPole-v0")
env = env.unwrapped
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.n

print("state dimensions : {} \naction dimensions : {}".format(s_dim, a_dim))

agent = Agent(0.001, 0.01, (s_dim,), a_dim)

#hyperparameters
games = 1000
render = True
thr_step = 500

rewards = []

for game in range(games):
    s = env.reset()
    steps = 0
    total_reward = 0
    for i in range(thr_step):
        a, prob = agent.choose_action(s)
        s_, r, done, _ = env.step(a)
        agent.remember(s,a,s_,r,done,prob)
        total_reward+=r
        steps+=1
        if render:
            env.render()
        s = s_
        if done:
            break
    #print(agent.memory.actions)
    agent.learn(40)
    rewards.append(total_reward)
    rewards = rewards[-1000:]
    if game%50==0:
        print("episode : {} | reward : {} | average reward : {}".format(game, total_reward,
                                                                         sum(rewards)/len(rewards)))
