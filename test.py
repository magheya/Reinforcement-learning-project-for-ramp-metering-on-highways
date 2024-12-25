import os
import traci
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000
MAX_STEPS = 100
STATE_BINS = 5  # Adjust the number of state bins as needed

# Define the SUMO environment
class SumoRampEnv:
    def __init__(self):
        self.sumoCmd = [
            "sumo",
            "-c", "C:/Users/boume/OneDrive/Bureau/RL project/Reinforcement-learning-project-for-ramp-metering-on-highways-main/RL_project.sumocfg"
        ]
        self.actions = [0, 1, 2]  # Green, Yellow, Red
        self.state_size = 6  # State dimensions
        self.action_size = len(self.actions)
        traci.start(self.sumoCmd)

    def reset(self):
        traci.close()
        traci.start(self.sumoCmd)
        return self.get_state()

    def step(self, action):
        if action == 0:
            traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green
        elif action == 1:
            traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow
        else:
            traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red

        traci.simulationStep()
        next_state = self.get_state()
        reward = self.calculate_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0
        return next_state, reward, done

    def get_state(self):
        try:
            highway_density = traci.edge.getLastStepVehicleNumber("highway_entry") / 3
            ramp_density = traci.edge.getLastStepVehicleNumber("ramp_entry")
            avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")
            queue_length = traci.edge.getLastStepHaltingNumber("ramp_entry")
            traffic_light_phase = traci.trafficlight.getPhase("ramp_metering_tl")
            phase_duration = traci.trafficlight.getPhaseDuration("ramp_metering_tl")

            return np.array([
                highway_density, ramp_density, avg_speed, queue_length,
                traffic_light_phase, phase_duration
            ])
        except traci.exceptions.TraCIException as e:
            print(f"Error during state retrieval: {e}")
            return np.zeros(self.state_size)

    def calculate_reward(self):
        highway_congestion = traci.edge.getLastStepHaltingNumber("highway_entry")
        ramp_queue = traci.edge.getLastStepHaltingNumber("ramp_entry")
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")

        alpha = 0.5
        beta = 0.5
        gamma = 1.0

        reward = -(alpha * ramp_queue) - (beta * highway_congestion) + (gamma * avg_speed)

        phase_switch_penalty = -0.1 if traci.trafficlight.getPhase("ramp_metering_tl") == 1 else 0.0

        return reward + phase_switch_penalty

    def close(self):
        traci.close()


# Define the Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}

    def get_q_value(self, state, action):
        state = tuple(state)
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        state = tuple(state)
        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        current_q = self.get_q_value(state, action)
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)
        self.q_table[(state, action)] = new_q

    def act(self, state):
        if np.random.rand() < EPSILON:
            return np.random.choice(self.action_size)
        return np.argmax([self.get_q_value(state, a) for a in range(self.action_size)])


# Train the Q-Learning agent
def main():
    global EPSILON  # Declare EPSILON as global here

    env = SumoRampEnv()
    agent = QLearningAgent(env.state_size, env.action_size)

    rewards_per_episode = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}")

        # Decay epsilon
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

    # Plot rewards
    plt.plot(rewards_per_episode)
    plt.title('Total Reward Across Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
