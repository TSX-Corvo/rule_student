from typing import Dict, Tuple
import gym
from gym import spaces
import numpy as np
import random


emotions = ["anger", "surprise", "disgust", "enjoyment", "fear", "sadness"]

categories = ["literature", "vocabulary", "idioms", "grammar"]

# Define rules
rules = {
    ("sadness", "grammar"): {
        "correct_chance": 0.7,
        "next_emotion": {"happiness": 0.8, "fear": 0.2},
    },
    ("fear", "literature"): {
        "correct_chance": 0.6,
        "next_emotion": {"disgust": 0.7, "surprise": 0.3},
    },
    ("anger", "idioms"): {
        "correct_chance": 0.8,
        "next_emotion": {"anger": 0.1, "disgust": 0.6, "surprise": 0.3},
    },
    ("enjoyment", "vocabulary"): {
        "correct_chance": 0.9,
        "next_emotion": {"enjoyment": 0.5, "surprise": 0.5},
    },
    ("surprise", "grammar"): {
        "correct_chance": 0.5,
        "next_emotion": {"surprise": 0.4, "enjoyment": 0.6},
    },
    ("disgust", "idioms"): {
        "correct_chance": 0.7,
        "next_emotion": {"disgust": 0.3, "anger": 0.7},
    },
}


class StudentEnv(gym.Env):
    current_emotion: int

    def __init__(self):
        # Define action and observation space
        # Action space: Categories of questions to ask
        self.action_space = spaces.Discrete(len(categories))

        # Observation space: Emotions as states
        self.observation_space = spaces.Discrete(len(emotions))

        # Define initial state
        self.current_emotion = np.random.choice(len(emotions))

        # Define other parameters as needed

    def reset(self):
        # Reset the environment to the initial state
        self.current_emotion = np.random.choice(len(emotions))
        return self.current_emotion

    def step(self, action):
        # Execute one time step within the environment

        # Apply stochastic rules to determine correctness and new emotion
        correct, next_emotion = self.apply_rules(action)

        # Update the current emotion based on the rules or any other logic
        self.current_emotion = emotions.index(next_emotion)

        # Return the new state, reward, and other info
        return self.current_emotion, correct, {}, {}

    def apply_rules(self, category: str) -> Tuple[bool, str]:
        emotion = emotions[self.current_emotion]

        rule: Dict[Tuple[str, str], dict] = rules.get((emotion, category), None)
        if rule is not None:
            correct_chance: float = rule["correct_chance"]
            next_emotion_probs: Dict[str, float] = rule["next_emotion"]

            # Determine correctness based on the chance
            correct = random.random() < correct_chance

            # Determine the next emotion based on probabilities
            next_emotion = random.choices(
                list(next_emotion_probs.keys()),
                weights=list(next_emotion_probs.values()),
            )[0]

            return correct, next_emotion
        else:
            # Default case if there's no specific rule defined
            return random.choice([True, False]), random.choice(emotions)
