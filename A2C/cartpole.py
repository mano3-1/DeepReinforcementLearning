import gym
import numpy as np
from agent_discrete import A2C
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')
print(env.action_space, env.observation_space)
agent = A2C(0.99, 0.001, 0.01, (4,), 2)

num_episodes = 2000

score_history = []
episodes_list = []
for episode in range(num_episodes):
    done = False
    score = 0
    state = env.reset()
    while not done:
        if episode>2000:
            env.render()
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.learn(state, reward, state_, done)
        state = state_
        score+=reward
    score_history.append(score)
    episodes_list.append(episode)
    avg_score = np.mean(score_history[-100:])
    if episode%50==0:
        print("episode : {} | average score : {}".format(episode, avg_score))
      
plt.plot(episodes_list, score_history)
plt.savefig("cartpole.jpg")
