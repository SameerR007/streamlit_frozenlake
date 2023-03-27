import streamlit as st
import numpy as np
import gym # pip intall gym
import random


total_episodes=st.number_input("Input the number of games computer can play to train itself. Remember, higher number of training iterations gives higher accuracy in the final test run",1)
total_episodes=int(total_episodes)
if st.checkbox("Select"):
    env = gym.make('FrozenLake-v1', is_slippery = False)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    qtable = np.zeros((state_space_size, action_space_size))
    learning_rate = 0.2 # 0-1
    max_steps = 100
    gamma = 0.99
    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    decay_rate = 0.001
    rewards = []
    for episode in range(total_episodes):
        state = env.reset()[0]
        step = 0
        done = False
        total_rewards = 0
        
        for step in range(max_steps):
            if random.uniform(0,1) > epsilon:
                action = np.argmax(qtable[state,:]) #Exploit
            else:
                action = env.action_space.sample() #Explore
                
            new_state, reward, done,trunc, info = env.step(action)
            
            max_new_state = np.max(qtable[new_state,:])
            qtable[state,action] = qtable[state,action] + learning_rate*(reward+gamma*max_new_state-qtable[state,action])
            
            total_rewards += reward
            
            state = new_state
            if done:
                break
            
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(total_rewards)

    st.markdown("The accuracy of the model after {} training iterations is {}. Which means that for every 100 games the computer plays, it will succeed in {} games.".format(total_episodes,round(sum(rewards)/total_episodes,2),round(sum(rewards)/total_episodes,2)*100))

    if st.button("Computer Plays"):
        env = gym.make('FrozenLake-v1', is_slippery = False,render_mode='rgb_array')
        env.reset()

        for episode in range(1):
            state = env.reset()[0]
            step = 1
            done = False
                
            c=1
            for step in range(1,max_steps+1):
                if random.uniform(0,1) > epsilon:
                    action = np.argmax(qtable[state,:]) #Exploit
                else:
                    action = env.action_space.sample() #Explore
                new_state, reward, done,trunc, info = env.step(action)
                st.text("Step-"+str(c))
                c=int(c)+1
                img = env.render()
                    
                st.image(img)
                    
                if done:
                    break
                state = new_state
                if step==max_steps:
                    st.text("Maximum steps reached.")        
        env.close()
