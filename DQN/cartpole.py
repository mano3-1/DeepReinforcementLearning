import gym
from matplotlib import pyplot as plt
import cv2
import numpy as np
from DQNAgent import Agent


def decay(value ,time_step ,thr ,mode = 'min' ,constant=0.001):
    if mode=='max':
        constant = -1*constant
    value = value*np.exp(-constant*time_step)
    if mode=='max':
        return min(value,thr)
    else:
        return max(value,thr)

env = gym.make('CartPole-v0')
#best params
#gamma = 0.999 ,epsilon = 1.0, lr = 0.01, max_mem_size=10000, fc1 = 256, gc2 = 128,n_games=500, 
agent = Agent(gamma=0.8, epsilon=0.3, lr=0.001, tau = 0.3, max_mem_size=10000,
              input_dims = (4,), batch_size=128, n_actions=2, fc = True, fc1 = 256,fc2 = 128)

#store start values for decay
gamma_start = agent.gamma
epsilon_start = agent.epsilon
tau_start = agent.tau

scores, eps_history = [],[]
n_games = 1000

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_,reward,done,info = env.step(int(action))
        #env.render() uncomment if u want to see how agent is getting trained,but rendering at every step increases the time taken to get trained
        agent.store_actions(observation,action,observation_,reward,done)
        agent.learn()
        observation = observation_
        score+=reward
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    if i%50 == 0:
        print("episode : {} | score : {} | average score :{} | epsilon : {} | gamma : {} | tau : {}".format(
                                                                            i,score,avg_score,agent.epsilon, agent.gamma, agent.tau))
    #decay tau,gamma,epsilon
    agent.tau = decay(tau_start,i+1,0.01, constant = 0.001)
    agent.gamma = decay(gamma_start,i+1,0.999,mode='max', constant = 0.001)
    agent.epsilon = decay(epsilon_start,i+1,0.001,constant=0.001)

    if i%2==0:
        agent.train_targetnet()
        
T.save(agent.Q_eval.state_dict(),r'D:\AI\REINFORCEMENT_LEARNING\DQN\CartPole.pt')
x = [i+1 for i in range(n_games)]
plt.plot(x,scores)
plt.scatter(x,eps_history)
plt.savefig(r"D:\AI\REINFORCEMENT_LEARNING\DQN\lCurve_cartpole.png")  