import traci

# Path to your SUMO configuration file
sumoConfigFile = "RL_project.sumocfg"

def main():
    # Start the SUMO simulation
    sumoBinary = "sumo-gui"  # Use "sumo" if you don't need the GUI
    sumoCmd = [sumoBinary, "-c", sumoConfigFile]
    traci.start(sumoCmd)

    # Simulation loop
    step = 0
    while step < 3600:  # Run for 3600 simulation seconds
        traci.simulationStep()  # Advance simulation by one step

        # Example: Get queue length on the ramp
        ramp_queue_length = len(traci.edge.getLastStepVehicleIDs("ramp_entry"))
        print(f"Step {step}: Queue length on ramp: {ramp_queue_length}")

        # Control the traffic light based on the queue length
        if ramp_queue_length > 10:
            traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green for ramp
        else:
            traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red for ramp

        step += 1

    # Close the simulation
    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    main()
