# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

# 309505018 ChunTing Kuo

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

env = gym.make("Blackjack-v0")


def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """

    # value function
    V = defaultdict(float)

    # print(V)
    # exit(0)
    '''
    Gt = sample discounted reward start from t
    gamma**0 * R[t] + gamma**1 * R[t+1] + ... + gamma**(Ti-2) * R[Ti-1] + gamma**(Ti-1) * R[Ti]
    GT = gamma**0 * r[T]
    G(T-1) = r[T-1] + gamma**1 * r[T] = r[T-1] + gamma * GT
    G(T-2) = r[T-2] + gamma * G(T-1) = r[T-2] + gamma * r[T-1] + gamma^2 * r[T]
    '''

    ##### FINISH TODOS HERE #####
    def generate_episode():
        states, actions, rewards = [], [], []
        observation = env.reset()
        while True:
            states.append(observation)
            action = policy(observation)
            actions.append(action)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        return states, actions, rewards

    N = defaultdict(int)

    for k in range(num_episodes):
        states, _, rewards = generate_episode()
        returns = 0
        print('k: ', k)
        for t in range(len(states) - 1, -1, -1):
            # print(t)
            R = rewards[t]
            S = states[t]
            returns += R
            if S not in states[:t]:  # first visit
                N[S] += 1
                V[S] += (returns - V[S]) / N[S]
                # print('states[:t]: ', states[:t])
                # print('t: ', t)
            returns *= gamma
            # print('returns: ', returns)

    #############################

    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    # value function
    V = defaultdict(float)
    alpha = 0.01

    ##### FINISH TODOS HERE #####
    for k in range(num_episodes):

        state = env.reset()  # Reset the environment

        # Initialize variables
        reward = 0
        terminated = False

        while not terminated:
            action = policy(state)
            # action = env.action_space.sample()

            # Take action
            next_state, reward, terminated, info = env.step(action)

            # Recalculate V[s]
            V[state] = V[state] + alpha * (
                reward + gamma * V[next_state] - V[state])
            print(V[state])

            state = next_state

    #############################

    return V


def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2,
                                  np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2,
                                np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=matplotlib.cm.coolwarm,
            vmin=-1.0,
            vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    # '''
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")
    # '''
    # '''
    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")
    # '''