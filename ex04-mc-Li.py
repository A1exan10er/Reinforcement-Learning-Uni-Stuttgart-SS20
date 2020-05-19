import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('Blackjack-v0')

def draw_plot(V1,V2):
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax = Axes3D(fig1)
    bx = Axes3D(fig2)
    
    X = np.arange(12,22) #player_sum
    Y = np.arange(1,11) #dealer_card
    X, Y = np.meshgrid(X, Y)
    Z1 = V1.T #row:player_sum  column:dealer_card
    Z2 = V2.T
    
    ax.plot_surface(X, Y, Z1,cmap='coolwarm')
    ax.set_zlim(-1, 1)

    bx.plot_surface(X, Y, Z2,cmap='coolwarm')
    bx.set_zlim(-1,1)
    
    plt.show()

def evaluate(episode,stick_point):
    V1 = np.zeros((10,10))#with usable ace row:dealer's card(1-10); column:current sum(12-21)
    V2 = np.zeros((10,10))#without usable ace
    Appearance1 = np.zeros((10,10))#count the appearance of states
    Appearance2 = np.zeros((10,10))
    for i in range(episode):
        #print("episode:",i+1)
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        dealer_card = obs[1]
        usable_ace = obs[2]
        done = False
        player_sum = []
        rewards = []
        Returns1 = np.zeros((10,10))#with usable ace
        Returns2 = np.zeros((10,10))
        while not done:
            player_sum.append(obs[0])
            #print("observation:", obs)
            if obs[0] >= stick_point:
                #print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                obs, reward, done, _ = env.step(1)
            rewards.append(reward)
            #print("reward:", reward)
            #print("")
            if usable_ace:
                for j in range(len(player_sum)):
                    Returns1[dealer_card-1][player_sum[j]-12] = sum(rewards[j:])
                    Appearance1[dealer_card-1][player_sum[j]-12] += 1
                V1 += Returns1
            else:
                for j in range(len(player_sum)):
                    Returns2[dealer_card-1][player_sum[j]-12] = sum(rewards[j:])
                    Appearance2[dealer_card-1][player_sum[j]-12] += 1
                V2 += Returns2
    V1 /= Appearance1
    V2 /= Appearance2
    return V1,V2

def find_optimal(episode):
    V1_opt = -1*np.ones((10,10))
    V2_opt = -1*np.ones((10,10))
    policies1 = np.zeros((10,10))
    policies2 = np.zeros((10,10))
    allpolicies = [12,13,14,15,16,17,18,19,20,21]
    for policy in allpolicies:
        V1,V2 = evaluate(episode,policy)
        for i in range(np.shape(V1)[0]):
            for j in range(np.shape(V1)[1]):
                comparison = (policies1[i][j],policy)
                if V1_opt[i][j]<V1[i][j]:
                    V1_opt[i][j] = V1[i][j]
                    policies1[i][j] = policy
                
                comparison = (policies2[i][j],policy)
                if V2_opt[i][j]<V2[i][j]:
                    V2_opt[i][j] = V2[i][j]
                    policies2[i][j] = policy
    
    return V1_opt,V2_opt,policies1, policies2

def interpret(policies):
    for i in range(np.shape(policies)[0]):
        for j in range(np.shape(policies)[1]):
            if policies[i][j] > j+12:
                policies[i][j] = 1  #1 for hit
            else:
                policies[i][j] = 0
    return policies

def formating(string):
    string = string.replace("\n ","\\\ ")
    string = string.replace(" ","&")
    string = string.replace("["," ")
    string = string.replace("]"," ")
    return string

def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20

    #obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    #done = False
    episode = 10000
    
    
    
    #V1,V2 = evaluate(episode,20)
    #print("Value function with usable ace:",V1)
    #print("Value function without usable ace:",V2)
    #draw_plot(V1,V2)

    V1_opt,V2_opt,policies1, policies2 = find_optimal(episode)
    policies1 = interpret(policies1)
    policies2 = interpret(policies2)
    print("optimal policy for with usable ace:\n",policies1)
    print("value function for with usable ace:\n",V1_opt)
    print("optimal policy for without usable ace:\n",policies2)
    print("value function for without usable ace:\n",V2_opt)

    #format for latex
    #print("optimal policy for with usable ace:\n",formating(str(policies1)))
    #print("optimal policy for without usable ace:\n",formating(str(policies2)))

    

if __name__ == "__main__":
    main()
