import gym
from matplotlib import pyplot as plt
import cv2
from DQNAgent import Agent
import numpy as np


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[25:200]
    image = cv2.resize(image,(75,75))
    image = image[np.newaxis,:,:]
    #image = image.transpose((2,0,1))
    return image

def decay(value ,time_step ,thr ,mode = 'min' ,constant=0.001):
    if mode=='max':
        constant = -1*constant
    value = value*np.exp(-constant*time_step)
    if mode=='max':
        return min(value,thr)
    else:
        return max(value,thr)
env = gym.make("SpaceInvaders-v0")
agent = Agent(gamma=0.8, epsilon=0.3, lr=0.001, tau=0.3, max_mem_size=10000,
              input_dims = (4,75,75), batch_size=128, n_actions=6, fc = False, fc1 = 256,fc2 = 128)

#store start values for decay
gamma_start = agent.gamma
epsilon_start = agent.epsilon
tau_start = agent.tau

scores, eps_history = [], []
n_games = 200

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    past_frames = [np.zeros((1,75,75)) for i in range(4)]
    past_frames[-1] = preprocess(observation)
    while not done:
        state = np.concatenate(past_frames,axis=0)
        action = agent.choose_action(state) 
        observation_, reward, done, info = env.step(int(action))
        
        past_frames.append(preprocess(observation))
        past_frames = past_frames[1:]

        state_ = np.concatenate(past_frames)
        agent.store_actions(state,action,state_,reward,done)
        agent.learn()

        score+=reward


    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    if i%10==0:
        print("episode : {} | score : {} | average score :{} | epsilon : {} | gamma : {} | tau : {}".format(
                                                                            i,score,avg_score,agent.epsilon, agent.gamma, agent.tau))
    
    #decay tau,gamma,epsilon
    agent.tau = decay(tau_start,i+1,0.1)
    agent.gamma = decay(gamma_start,i+1,0.999,mode='max')
    agent.epsilon = decay(epsilon_start,i+1,0.001,constant=0.03)

    if i%2==0:
        agent.train_targetnet()