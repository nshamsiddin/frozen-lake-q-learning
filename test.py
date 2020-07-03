# https://deeplizard.com/learn/video/QK_PP_2KgGE

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

# create a q-table and fill with 0s
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning params

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.4
discount_rate = 0.99

# epsilon greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

rewards_all_episodes = []

print(q_table)

# Q-learning algorithm
for episode in range(num_episodes):
	state = env.reset()

	done = False
	rewards_current_episode = 0

	for step in range(max_steps_per_episode): 

		# Exploration-exploitation trade-off
		exploration_rate_threshold = random.uniform(0, 1)
		if exploration_rate_threshold > exploration_rate:
			action = np.argmax(q_table[state,:]) 
		else:
			action = env.action_space.sample()
		new_state, reward, done, info = env.step(action)

		# Update Q-table for Q(s,a)
		q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
			learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

		state = new_state
		rewards_current_episode += reward 

		if done == True:
			break
	# print(rewards_current_episode)
		# Exploration rate decay
	if episode in [1, 100, 1000, 5000, 10000, 50000, 100000]:
		print('\nQ-table for episode #' +  str(episode))
		print(q_table)

	exploration_rate = min_exploration_rate + \
		(max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
		
	rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print(q_table)

print("\n********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
	print(count, "\t", str(sum(r/1000)))
	count += 1000
