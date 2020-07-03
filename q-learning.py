# frozen-lake/q-learning.py
# https://deeplizard.com/learn/video/QK_PP_2KgGE

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

class QLearningAgent:
	
	def __init__(self, num_episodes = 10000, max_steps_per_episode = 100, learning_rate = 0.1, 
				discount_rate = 0.99, exploration_rate = 1, max_exploration_rate = 1, 
				min_exploration_rate = 0.01, exploration_decay_rate = 0.01):
		# create a q-table and fill with 0s
		self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
		self.num_episodes = num_episodes
		self.max_steps_per_episode = max_steps_per_episode

		self.learning_rate = learning_rate
		self.discount_rate = discount_rate

		self.exploration_rate = exploration_rate
		self.max_exploration_rate = max_exploration_rate
		self.min_exploration_rate = min_exploration_rate
		self.exploration_decay_rate = exploration_decay_rate

		self.rewards_all_episodes = []
		
		print(self.q_table)
	
	def _learn(self):
		# Q-learning algorithm
		for episode in range(self.num_episodes):
			state = env.reset()

			done = False
			self.rewards_current_episode = 0

			for step in range(self.max_steps_per_episode): 

				# Exploration-exploitation trade-off
				exploration_rate_threshold = random.uniform(0, 1)
				if exploration_rate_threshold > self.exploration_rate:
					action = np.argmax(self.q_table[state,:]) 
				else:
					action = env.action_space.sample()
				new_state, reward, done, info = env.step(action)
				# Update Q-table for Q(s,a)
				self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + \
					self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state, :]))
				state = new_state
				self.rewards_current_episode += reward 

				if done == True:
					break
			# print(rewards_current_episode)
				# Exploration rate decay
			if episode in [1, 100, 1000, 5000, 10000, 50000, 100000]:
				print('\nQ-table for episode #' +  str(episode))
				print(self.q_table)

			exploration_rate = self.min_exploration_rate + \
				(self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate*episode)
			self.rewards_all_episodes.append(self.rewards_current_episode)

	def _stats(self):
		# Calculate and print the average reward per thousand episodes
		rewards_per_thousand_episodes = np.split(np.array(self.rewards_all_episodes),self.num_episodes/1000)
		count = 1000

		print(self.q_table)

		print("\n********Average reward per thousand episodes********\n")
		for r in rewards_per_thousand_episodes:
			print(count, ": ", str(sum(r/1000)))
			count += 1000

def main():
	agent = QLearningAgent()
	agent._learn()
	agent._stats()


if __name__ == "__main__":
    main()