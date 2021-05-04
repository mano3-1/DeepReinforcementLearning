import gym
import numpy as np
from agent import DDPG
from matplotlib import pyplot as plt

def normalize(state):
    return state/8
env = gym.make('Pendulum-v0')
env = env.unwrapped
print(env.action_space, env.observation_space)
agent = DDPG(0.0005, 0.005, 0.9, (3,), (1,), 2)
n_games = 10000
scores = []
eps_history = []
prev_max_avg_score = 0
for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    steps = 0
    while not done:
        action = agent.choose_action(normalize(observation))
        observation_,reward,done,info = env.step(np.clip(action,-2,2).reshape(-1))
        #env.render()
        agent.store(normalize(observation),action,normalize(observation_),reward,done)
        agent.learn()
        observation = observation_
        score+=reward
        steps+=1
        if steps>500:
            done = True
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    if i%5 == 0:
        print("episode : {} | score : {} | average score :{} | steps : {}".format(i,score,avg_score,steps))
    if i%2==0:
        agent.update_target_nets()
    if prev_max_avg_score<avg_score:
        agent.save_weights()
        prev_max_avg_score = avg_score