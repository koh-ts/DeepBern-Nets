import json
import glob
import os
import numpy as np
import re

from datasets import StaliroDataset

def data_process(data_file):
    # Load the data from /home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/20241105_133445.json 
    # and extract "init_cond", "samples", "states", and "times"
    # "samples" is a 4 types x 4 control actions for 30 sec simulation, where the same control action is applied for 7.5 seconds each
    # so [0, 7] is the first control action, [8, 14] is the second control action, [15, 22] is the third control action, and [23, 29] is the fourth control action
    # "states" is 31 timesteps from 0 to 30 inclusive
    # "init_cond" is the initial condition of the system, which is the height of the tank at t=0
    # the pair of one of the "states" and "samples" are the input to our NN, and the output is the next time step's state

    # Example data file path
    # data_file = "/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data/20241105_133445.json"
    
    with open(data_file, 'r') as f:
        data_list = json.load(f)  # This is now a list of dictionaries
    
    # Initialize lists to store all input-output pairs
    all_inputs = []
    all_outputs = []
    
    # Process each dictionary in the list
    for idx, data in enumerate(data_list):
        print(f"Processing data entry {idx + 1}/{len(data_list)}")
        
        # Extract the required fields from this entry
        samples = data["samples"]  # Control actions
        states = data["states"]    # System states (31 timesteps)
        times = data["times"]      # Time values
                
        # Function to get control action for a given timestep
        def get_control_action(timestep):
            if timestep < 8:
                return samples[0::4]
            elif timestep < 15:
                return samples[1::4]
            elif timestep < 23:
                return samples[2::4]
            elif timestep < 30:
                return samples[3::4]
            return None
        
        # Create input-output pairs for this entry
        entry_inputs = []
        entry_outputs = []
        
        # For each timestep (except the last one), create input-output pair
        for t in range(len(states) - 1):  # 0 to 29 (30 pairs total)
            # Input: current state + control action for this timestep
            current_state = states[t][0]
            control_action = get_control_action(t)
            
            if control_action is not None:
                # Combine current state and control action as input
                input_vector = [current_state] + control_action
                entry_inputs.append(input_vector)
                
                # Output: next timestep's state
                next_state = states[t + 1][0]
                entry_outputs.append(next_state)
        
        # Add this entry's data to the overall lists
        all_inputs.extend(entry_inputs)
        all_outputs.extend(entry_outputs)
        
        print(f"  Entry {idx + 1}: {len(entry_inputs)} input-output pairs")
    
    # Convert to numpy arrays
    inputs = np.array(all_inputs)
    outputs = np.array(all_outputs)
    
    print(f"\nTotal number of entries processed: {len(data_list)}")
    print(f"Total number of input-output pairs: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Sample input (state + control): {inputs[0]}")
    print(f"Sample output (next state): {outputs[0]}")
    
    # Optional: Save the processed data
    processed_data = {
        "inputs": inputs.tolist(),
        "outputs": outputs.tolist(),
        "num_entries": len(data_list),
        "pairs_per_entry": len(all_inputs) // len(data_list) if len(data_list) > 0 else 0
    }
    
    return inputs.tolist(), outputs.tolist()

def iterate_json_files(directory):
    json_files = []
    inputs = []
    outputs = []
    i = 0
    for file_path in glob.glob(os.path.join(directory, "*.json")):
        i += 1
        if i > 5:
            print("Reached 5 files, stopping iteration.")
            break
        print(f"Processing file: {file_path}")
        if "all_data" in file_path:
            continue
        input_, output_ = data_process(file_path)
        inputs += input_
        outputs += output_
    # Save the processed data to a file in dynamicsNN directory
    output_file = os.path.join(directory+"/dynamicsNN", "processed_data.json")
    with open(output_file, 'w') as f:
        json.dump({"inputs": inputs, "outputs": outputs}, f, indent=2)
    print("all done")

if __name__ == "__main__":
    directory = "/home/koh/work/matiec_rampo/examples/tankcontrol_flowrate/data"
    iterate_json_files(directory)
