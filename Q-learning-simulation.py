import numpy as np
import traci
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size  # Now includes Yellow
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore: random action
        return np.argmax(self.q_table[state])  # Exploit: best action

    def learn(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        # Decay epsilon to favor exploitation over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def get_state():
    """Map traffic conditions to a discrete state."""
    ramp_queue = len(traci.edge.getLastStepVehicleIDs("ramp_entry"))
    highway_speed = traci.edge.getLastStepMeanSpeed("highway_entry")

    # Example: Define states based on queue length and highway speed
    if ramp_queue > 10:
        if highway_speed < 20:
            return 0  # High queue, low speed
        else:
            return 1  # High queue, high speed
    else:
        if highway_speed < 20:
            return 2  # Low queue, low speed
        else:
            return 3  # Low queue, high speed


def main():
    sumoBinary = "sumo-gui"  # Use "sumo" for non-GUI mode
    sumoCmd = [sumoBinary, "-c", "RL_project.sumocfg"]

    # Initialize the Q-learning agent
    state_size = 4  # Four discrete states as defined in `get_state`
    action_size = 3  # Actions: 0 = Green, 1 = Yellow, 2 = Red
    agent = QLearningAgent(state_size, action_size)

    episodes = 100  # Number of episodes for training
    reward_history = []  # Store rewards for each episode

    # Start SUMO GUI once
    traci.start(sumoCmd)

    for episode in range(episodes):

        print(f"Starting Episode {episode + 1}")
        traci.load(["-c", "RL_project.sumocfg"])  # Reset the simulation
    
        total_reward = 0

        state = get_state()  # Initialize the starting state
        prev_action = -1  # To track the previous action

        for step in range(3600):  # Simulate for 3600 seconds
            action = agent.choose_action(state)

            # Apply traffic light logic
            if action == 0:  # Green light
                if prev_action == 2:  # If transitioning from red
                    traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow phase
                    traci.simulationStep()
                traci.trafficlight.setPhase("ramp_metering_tl", 0)
            elif action == 1:  # Yellow light
                traci.trafficlight.setPhase("ramp_metering_tl", 1)
            elif action == 2:  # Red light
                if prev_action == 0:  # If transitioning from green
                    traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow phase
                    traci.simulationStep()
                traci.trafficlight.setPhase("ramp_metering_tl", 2)

            # Advance simulation
            traci.simulationStep()

            # Get next state and reward
            next_state = get_state()
            reward = (
                -len(traci.edge.getLastStepVehicleIDs("ramp_entry"))  # Penalize long queues
                + traci.edge.getLastStepMeanSpeed("highway_entry")  # Reward high speed
            )
            total_reward += reward

            # Update the Q-table
            agent.learn(state, action, reward, next_state)

            # Transition to the next state
            state = next_state
            prev_action = action

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        reward_history.append(total_reward)  # Track rewards

    traci.close()

    # Save the Q-table for future use
    np.save("q_table.npy", agent.q_table)
    print("Training complete. Q-table saved.")

    # Plot reward history
    plt.plot(reward_history)
    plt.title("Total Reward Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
