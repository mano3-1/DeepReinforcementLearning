import gym
import numpy as np
from agent_continuous import A2C
from matplotlib import pyplot as plt


env = gym.make('Pendulum-v0')
print(env.action_space, env.observation_space)
agent = A2C(0.99, 0.001, 0.01, (3,), 1)

num_episodes = 10000
score_history = []
episodes_list= []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action.reshape(1))
        agent.learn(state, reward, state_, done)
        score+=reward
        state = state_
    score_history.append(score)
    episodes_list.append(episode)
    avg_score = np.mean(score_history[-100:])
    if episode%50==0:
        print("episode : {} | average score : {}".format(episode, avg_score))