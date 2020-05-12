import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n #For 4x4 square, counting each position from left to right, top to bottom
n_actions = env.action_space.n #0 left 1 down 2 right 3 up

'''
SFFF
FHFH
FFFH
HFFG'''

def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    Theta = 1e-8 # make the while start
    step = 0
    while Theta >= theta:
        step += 1
        Theta = 0
        policy = []
        for i in range(n_states):
            v_state = V_states[i]
            v_function_list = []
            for j in range(n_actions):
                a = env.P[i][j]
                sum = 0
                for k in range(len(a)):
                    p = a[k][0]
                    n_state = a[k][1]
                    r = a[k][2]
                    is_terminal = a[k][3]
                    print("from state {} taking action {} leads to state {} with possibility {} and reward {}".format(i,j,n_state,p,r))
                    print("Reach the terminal:{}".format(is_terminal))
                    sum += p * (r + gamma * n_state)
                v_function_list.append(sum) # This list contains all returns for every action from state i with action as index
                
            if i not in [5,7,11,12,15]:
                V_states[i] = max(v_function_list)
            policy.append(np.argmax(v_function_list))
            print(v_function_list)
            Theta = max(Theta, abs(v_state - V_states[i]))
    print("It takes {} steps to converge".format(step))
    print("The optimal value for all states are:")
    print(V_states)
    return policy
        


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
