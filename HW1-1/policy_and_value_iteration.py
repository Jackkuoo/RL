# Spring 2021, IOC 5269 Reinforcement Learning
# HW1, partI: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    # 1. Initialize value function
    v_k = np.zeros(num_spaces)

    # 2. Get transition prob. and reward
    R, P = get_rewards_and_transitions_from_env(env)  # (500, 6, 400)
    R = np.max(R, axis=2)

    # 3. Iterate and improve value
    for k in range(max_iterations):
        # for each state s
        v_k_next = np.zeros(num_spaces)
        new_policy = np.zeros(num_spaces)
        for s in range(num_spaces):
            a = policy[s]

            vs = R[s, :] + gamma * P[s, :, :] @ v_k
            a = np.argmax(vs)
            v_k_next[s] = vs[a]

            # 4. Derive the optimal policy using V
            policy[s] = a

        # print((v_k_next - v_k))
        if np.linalg.norm(v_k_next - v_k) <= eps:
            break
        v_k = v_k_next

    
    #############################
    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    # 1. Initialize value function to zeros
    V = np.zeros(num_spaces)

    # 2. Get transition prob. and reward
    R, P = get_rewards_and_transitions_from_env(env)

    # 3. Iterate and imporve the policy
    def policy_eval(policy):
        nonlocal num_actions, num_spaces, R, P
        # Initialize thel value function
        v_k = np.zeros(num_spaces)
        # Find value function
        while True:
            v_k_next = np.zeros(num_spaces)
            for s in range(num_spaces):
                a = policy[s]
                arg = np.argmax(abs(R[s, a, :]))
                v_k_next[s] = R[s, a, arg] + gamma * P[s, a, :] @ v_k
            if np.linalg.norm(v_k_next - v_k) <= eps:
                break
            v_k = v_k_next
        return v_k

    for k in range(max_iterations):
        v_k = policy_eval(policy)
        Q = np.sum(R, axis=2) + gamma * P @ v_k  #(500,6)

        changed = False
        for s in range(num_spaces):
            a = np.argmax(Q[s, :])
            if a != policy[s]:
                changed = True
            policy[s] = a
        if changed == False:
            break


    #############################

    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)
    



