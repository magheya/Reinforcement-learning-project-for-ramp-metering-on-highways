import traci
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input


# RL Agent for traffic light control
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))  # Using Input layer to specify input shape
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values)  # Best action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Retrieve state information
def get_state():
    return [
        len(traci.edge.getLastStepVehicleIDs("ramp_entry")),  # Queue length on ramp
        len(traci.edge.getLastStepVehicleIDs("highway_entry")),  # Queue length on highway
        traci.edge.getLastStepOccupancy("ramp_entry"),  # Vehicle density on ramp
        traci.edge.getLastStepOccupancy("highway_entry"),  # Vehicle density on highway
        traci.edge.getLastStepMeanSpeed("highway_entry"),  # Average speed on highway
    ]

# Main simulation logic
def run_episode(agent, sumoCmd, sumoConfigFile, episode, num_episodes, rewards_list):
    traci.start(sumoCmd)  # Start SUMO simulation connection
    traci.simulationStep()  # Take an initial step to start the simulation

    # Reset the simulation manually for the new episode
    traci.load([sumoConfigFile])  # Reload SUMO configuration
    traci.simulationStep()  # Advance one step to start the simulation

    # Reset all vehicles
    for vehicle_id in traci.vehicle.getIDList():
        traci.vehicle.remove(vehicle_id)

    # Reset traffic light state to initial (Green for ramp)
    traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green for ramp (Phase 0)
    
    state = get_state()  # Get initial state
    state = np.reshape(state, [1, agent.state_size])

    total_reward = 0  # Keep track of the total reward for the episode
    rewards = []  # List to store rewards for each step
    for step in range(3600):  # Simulate for 3600 seconds per episode
        action = agent.act(state)  # Choose an action (Green=0, Yellow=1, Red=2)

        # Set traffic light phase based on the action
        if action == 0:
            traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green for ramp
        elif action == 1:
            traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow for ramp
        elif action == 2:
            traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red for ramp

        traci.simulationStep()  # Advance simulation

        # Get next state and calculate reward
        next_state = get_state()  # Get next state
        next_state = np.reshape(next_state, [1, agent.state_size])
        reward = (
            -len(traci.edge.getLastStepVehicleIDs("ramp_entry"))  # Penalize ramp queue length
            + traci.edge.getLastStepMeanSpeed("highway_entry")  # Reward highway speed
        )
        done = step == 3599  # Check if episode ends
        agent.remember(state, action, reward, next_state, done)  # Store experience in memory
        state = next_state  # Move to the next state

        total_reward += reward  # Add reward to the total for this episode
        rewards.append(reward)  # Append reward to the list

        if done:
            print(f"Episode {episode+1}/{num_episodes}: Reward = {total_reward}")
            break  # End the current episode

    rewards_list.append(total_reward)
    traci.close()  # Close the simulation connection

def main():
    sumoConfigFile = "RL_project.sumocfg"
    sumoBinary = "sumo"  # Use "sumo" for command-line mode
    sumoCmd = [sumoBinary, "-c", sumoConfigFile, "--step-length", "0.1"]  # Increase step length for faster simulation

    # Initialize RL Agent
    state_size = 5  # Length of the state vector
    action_size = 3  # Three actions: Green, Yellow, and Red phases for the ramp
    agent = RLAgent(state_size, action_size)
    batch_size = 32

    # Train over multiple episodes
    num_episodes = 100  # Number of episodes to run
    rewards_list = []

    for episode in range(num_episodes):  # Train for 100 episodes
        run_episode(agent, sumoCmd, sumoConfigFile, episode, num_episodes, rewards_list)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # Train the model with replay memory

    # Save the trained model after all episodes
    agent.model.save("traffic_light_model.h5")
    print("Training complete. Model saved as traffic_light_model.h5")

    # Plot the evolution of rewards
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evolution of Rewards')
    plt.show()

    # Best policy explanation
    print("Best policy explanation:")
    for action in range(agent.action_size):
        print(f"Action {action}: {agent.model.predict(np.eye(agent.state_size)[action].reshape(1, -1))}")

if __name__ == "__main__":
    main()