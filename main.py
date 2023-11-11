import numpy as np

from StudentEnv import StudentEnv


class QLearningAgent:
    def __init__(
        self,
        num_emotions,
        num_categories,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_prob=0.2,
    ):
        self.num_emotions = num_emotions
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_emotions, num_categories))

    def select_action(self, current_emotion):
        # Epsilon-greedy strategy for action selection
        if np.random.rand() < self.exploration_prob:
            # Explore: choose a random action
            return np.random.choice(self.num_categories)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self.q_table[current_emotion, :]
            return np.argmax(q_values)

    def update_q_table(self, current_emotion, action, reward, next_emotion):
        # Q-value update using the Q-learning formula
        current_q_value = self.q_table[current_emotion, action]
        next_max_q_value = np.max(self.q_table[next_emotion, :])
        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * next_max_q_value - current_q_value
        )

        # Update Q-value in the table
        self.q_table[current_emotion, action] = new_q_value


# Example usage of Q-learning agent in your environment
num_emotions = 6
num_categories = 4
agent = QLearningAgent(num_emotions, num_categories)

env = StudentEnv()

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    # Reset the environment for a new episode
    current_emotion = env.reset()

    # Select an action based on the current state
    current_category = agent.select_action(current_emotion)
    correct, next_emotion = env.step(current_category)

    # Define a reward function based on correctness (you can customize this)
    reward = 1 if correct else 0

    # Update the Q-table based on the observed reward and the next state
    agent.update_q_table(current_emotion, current_category, reward, next_emotion)

    # Move to the next state
    current_emotion = next_emotion
