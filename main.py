import sys
import gym
from agent import QLearning
from gym.envs.registration import register

from env import StudentEnv


register(
    id="Student-v0",
    entry_point="env:StudentEnv",
    max_episode_steps=200,
)

# RobotBattery-v0, Taxi-v3, FrozenLake-v1, RobotMaze-v0
ENVIRONMENT = "Student-v0"


def train(env: StudentEnv, agent: QLearning, episodes: int):
    for _ in range(episodes):
        observation, _ = env.reset(seed=7)
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, "random")
            new_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, new_observation, reward, terminated)
            observation = new_observation


def play(env: StudentEnv, agent: QLearning):
    observation, _ = env.reset(seed=7)
    env.render()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_action(observation, "greedy")
        new_observation, reward, terminated, truncated, _ = env.step(action)
        agent.update(observation, action, new_observation, reward, terminated)
        observation = new_observation
        env.render()


if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    agent = QLearning(
        env.observation_space.n, env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.1
    )

    episodes = 100 if len(sys.argv) == 1 else int(sys.argv[1])

    train(env, agent, episodes)
    agent.render()
    env.close()

    env = gym.make(ENVIRONMENT)
    play(env, agent)

    env.close()
