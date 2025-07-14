import json
import glob
import os
import numpy as np
import re

def data_process(data_file):
    # Load the data from JSON file
    # "samples" contains 10 control values for 30 sec simulation
    # Each control value is applied for 3 seconds
    # so samples[0] is applied from 0-3s, samples[1] from 3-6s, etc.
    
    with open(data_file, 'r') as f:
        data_list = json.load(f)
    
    all_inputs = []
    all_outputs = []
    
    for idx, data in enumerate(data_list):
        print(f"Processing data entry {idx + 1}/{len(data_list)}")
        
        samples = data["samples"]  # 10 control values
        states = data["states"]
        times = data["times"]
        
        # 10 control periods, each lasting 3 seconds
        def get_control_action(timestep):
            if timestep >= len(times):
                return None
            time_value = times[timestep]
            
            if time_value < 3.0:
                return samples[0]    # First control value
            elif time_value < 6.0:
                return samples[1]    # Second control value
            elif time_value < 9.0:
                return samples[2]    # Third control value
            elif time_value < 12.0:
                return samples[3]    # Fourth control value
            elif time_value < 15.0:
                return samples[4]    # Fifth control value
            elif time_value < 18.0:
                return samples[5]    # Sixth control value
            elif time_value < 21.0:
                return samples[6]    # Seventh control value
            elif time_value < 24.0:
                return samples[7]    # Eighth control value
            elif time_value < 27.0:
                return samples[8]    # Ninth control value
            elif time_value < 30.0:
                return samples[9]    # Tenth control value
            return None
        
        entry_inputs = []
        entry_outputs = []
        
        # ダウンサンプリング設定
        downsample_factor = 5
        
        # 固定間隔ダウンサンプリング（切り替わり点は自動的に含まれる）
        for t in range(0, len(states) - 1, downsample_factor):
            # 現在の状態：[車速, エンジン回転数] 両方を使用
            current_state = states[t]  # [speed, RPM]
            control_action = get_control_action(t)  # 単一の制御値
            
            if control_action is not None:
                # 入力：[車速, エンジン回転数, 制御信号]
                input_vector = current_state + [control_action]
                entry_inputs.append(input_vector)
                
                # 出力：次の状態 [次の車速, 次のエンジン回転数]
                next_state = states[t + 1]  # [next_speed, next_RPM]
                entry_outputs.append(next_state)
        
        all_inputs.extend(entry_inputs)
        all_outputs.extend(entry_outputs)
        
        print(f"  Entry {idx + 1}: {len(entry_inputs)} input-output pairs")
    
    inputs = np.array(all_inputs)
    outputs = np.array(all_outputs)
    
    print(f"\nTotal processed: {len(data_list)} entries")
    print(f"Total pairs: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")  # [N, 3] (車速 + RPM + 1つの制御信号)
    print(f"Output shape: {outputs.shape}")  # [N, 2] (次の車速 + 次のRPM)
    
    return inputs.tolist(), outputs.tolist()

def iterate_json_files(directory):
    json_files = []
    inputs = []
    outputs = []
    i = 0
    for file_path in glob.glob(os.path.join(directory, "*.json")):
        i += 1
        if i > 3:
            print("Reached 3 files, stopping iteration.")
            break
        print(f"Processing file: {file_path}")
        if "all_data" in file_path:
            continue
        input_, output_ = data_process(file_path)
        inputs += input_
        outputs += output_
    # Save the processed data to a file in dynamicsNN directory
    output_file = os.path.join(directory+"/dynamicsNN_engine", "processed_data_reduced_2.json")
    with open(output_file, 'w') as f:
        json.dump({"inputs": inputs, "outputs": outputs}, f, indent=2)
    print("all done")

if __name__ == "__main__":
    directory = "/home/koh/work/matiec_rampo/examples/vehicle_engine/data_new"
    iterate_json_files(directory)
