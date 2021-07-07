# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C
# 309505018 Chunting Kuo

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        
class Policy(nn.Module):
	"""
	    Implement both policy network and the value network in one model
	    - Note that here we let the actor and value networks share the first layer
	    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
	    - Feel free to add any member variables/functions whenever needed
	    TODO:
	        1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
	        2. Random weight initialization of each layer
	"""
	def __init__(self):
	    super(Policy, self).__init__()
	    
	    # Extract the dimensionality of state and action spaces
	    self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
	    self.observation_dim = env.observation_space.shape[0]
	    self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
	    self.hidden_size = 128
	    
	    ########## YOUR CODE HERE (5~10 lines) ##########
	    # action network definition
	    self.policy_fc1 = torch.nn.Linear(self.observation_dim, self.hidden_size)
	    self.policy_fc2 = torch.nn.Linear(self.hidden_size, self.action_dim)
	    self.relu = torch.nn.ReLU()
	    self.softmax = torch.nn.Softmax(dim=-1)

	    # value network definition
	    self.value_fc1 = nn.Linear(self.observation_dim, 64)
	    self.value_fc2 = nn.Linear(64, 1)
	    
	    ########## END OF YOUR CODE ##########
	    
	    # action & reward memory
	    self.saved_actions = []
	    self.rewards = []

	def forward(self, state):
	    """
	        Forward pass of both policy and value networks
	        - The input is the state, and the outputs are the corresponding 
	          action probability distirbution and the state value
	        TODO:
	            1. Implement the forward pass for both the action and the state value
	    """
	    
	    ########## YOUR CODE HERE (3~5 lines) ##########
	    # state = torch.Tensor(state)
	    # action network
	    x = self.relu(self.policy_fc1(state))
	    action_prob = self.softmax(self.policy_fc2(x))

	    # value network
	    x = self.relu(self.value_fc1(state))
	    state_value = self.value_fc2(x)

	    ########## END OF YOUR CODE ##########

	    return action_prob, state_value


	def select_action(self, state):
	    """
	        Select the action given the current state
	        - The input is the state, and the output is the action to apply 
	        (based on the learned stochastic policy)
	        TODO:
	            1. Implement the forward pass for both the action and the state value
	    """
	    
	    ########## YOUR CODE HERE (3~5 lines) ##########
	    # state = torch.from_numpy(state).float()
	    # state = torch.Tensor(state)
	    action_probs, state_value = self.forward(state)

	    m = Categorical(action_probs) # same as multinomial
	    action = m.sample()

	    ########## END OF YOUR CODE ##########
	    
	    # save to action buffer
	    self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

	    return action.item(), m.log_prob(action), state_value


	def calculate_loss(self, gamma=0.99):
		"""
		    Calculate the loss (= policy loss + value loss) to perform backprop later
		    TODO:
		        1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
		        2. Calculate the policy loss using the policy gradient
		        3. Calculate the value loss using either MSE loss or smooth L1 loss
		"""
		
		# Initialize the lists and variables
		R = 0
		saved_actions = self.saved_actions
		policy_losses = [] 
		value_losses = [] 
		returns = []

		########## YOUR CODE HERE (8-15 lines) ##########


		########## END OF YOUR CODE ##########
		
		return loss

	def clear_memory(self):
	    # reset rewards and action buffer
	    del self.rewards[:]
	    del self.saved_actions[:]


def train(lr=0.01):
	'''
	    Train the model using SGD (via backpropagation)
	    TODO: In each episode, 
	    1. run the policy till the end of the episode and keep the sampled trajectory
	    2. update both the policy and the value network at the end of episode
	'''    
	
	# Instantiate the policy model and the optimizer
	model = Policy()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	
	# Learning rate scheduler (optional)
	scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
	
	# EWMA reward for tracking the learning progress
	ewma_reward = 0
	maxx = 0
	# run inifinitely many episodes
	for i_episode in count(1):
		# reset environment and episode reward
		state = env.reset()
		ep_reward = 0
		t = 0
		total_reward = 0
		episode_reward = []
		log_probs = []
		state_values = []
		R = 0
		# Uncomment the following line to use learning rate scheduler
		#scheduler.step()
		
		# For each episode, only run 9999 steps so that we don't 
		# infinite loop while learning
		
		########## YOUR CODE HERE (10-15 lines) ##########
		for t in range(1, 10000):

			# select action from policy
			action, a_log_prob, state_value = model.select_action(torch.from_numpy(state).float().unsqueeze(0))

			# take the action
			state, reward, done, _ = env.step(action)

			total_reward += reward
			ep_reward += reward
			episode_reward.append(reward)
			log_probs.append(a_log_prob)
			state_values.append(state_value)

			if done:
				break



		G = []
		for r in episode_reward:
		    R = r + 0.9 * R
		    G.insert(0, R)

		G = torch.tensor(G)

		# apply whitening
		# G = (G - G.mean()) / (G.std() + eps) # To have small values of Loss

		p_losses  = []
		v_losses = []

		for a_log_prob, state_value, R in zip(log_probs, state_values, G):
		    p_losses.append(-1 * a_log_prob * (R - state_value.item()))
		    v_losses.append(torch.nn.functional.mse_loss(state_value, torch.tensor([R]), reduction='mean'))

		optimizer.zero_grad()

		policy_loss = torch.stack(p_losses).sum()
		value_loss =  torch.stack(v_losses).sum()
		loss = policy_loss + value_loss
		# print(loss.item())
		loss.backward()
		optimizer.step()
		########## END OF YOUR CODE ##########
		    
		# update EWMA reward and log the results
		ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
		print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

		# check if we have "solved" the cart pole problem
		if ewma_reward > env.spec.reward_threshold:
		    torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
		    print("Solved! Running reward is now {} and "
		          "the last episode runs to {} time steps!".format(ewma_reward, t))
		    break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action, a_log_prob, state_value = model.select_action(torch.from_numpy(state).float())
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()

            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20  
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test('CartPole_0.01.pth')

