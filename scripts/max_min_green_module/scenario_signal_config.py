# configure_signal_phases_interactive.py

import os
import glob
import json
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
TFRECORD_DIR = '../waymo_movie_file'
CONFIG_FILENAME = 'scenario_phases.json'
STOP_LINE_PROXIMITY = 10.0 # meters (how close stop lines must be to be grouped)

def group_lanes_into_approaches(map_features):
    """Groups lanes into approaches based on the proximity of their stop lines."""
    lanes = [f for f in map_features if f.WhichOneof('feature_data') == 'lane' and len(f.lane.polyline) > 1]
    approaches = []
    processed_lane_ids = set()

    for i, lane1 in enumerate(lanes):
        if lane1.id in processed_lane_ids:
            continue
        
        current_approach = {
            'lane_ids': {lane1.id}, 
            'stop_line_pos': np.array([lane1.lane.polyline[0].x, lane1.lane.polyline[0].y])
        }
        processed_lane_ids.add(lane1.id)

        for j in range(i + 1, len(lanes)):
            lane2 = lanes[j]
            if lane2.id in processed_lane_ids:
                continue
            
            stop_line2_pos = np.array([lane2.lane.polyline[0].x, lane2.lane.polyline[0].y])
            if np.linalg.norm(current_approach['stop_line_pos'] - stop_line2_pos) < STOP_LINE_PROXIMITY:
                current_approach['lane_ids'].add(lane2.id)
                processed_lane_ids.add(lane2.id)
        
        approaches.append(current_approach)
    return approaches

def generate_config_frame(scenario, output_dir):
    """
    Generates a single frame showing all identified approaches with their index numbers.
    """
    approaches = group_lanes_into_approaches(scenario.map_features)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_facecolor('black')

    # Draw the base map
    for feature in scenario.map_features:
        feature_data = getattr(feature, feature.WhichOneof('feature_data'))
        if hasattr(feature_data, 'polyline') and len(feature_data.polyline) > 1:
            points = np.array([[p.x, p.y] for p in feature_data.polyline])
            ax.plot(points[:, 0], points[:, 1], color='gray', linewidth=1)

    # Draw and label each approach
    for idx, approach in enumerate(approaches):
        stop_line_pos = approach['stop_line_pos']
        ax.plot(stop_line_pos[0], stop_line_pos[1], 'o', color='cyan', markersize=10)
        ax.text(stop_line_pos[0], stop_line_pos[1] + 5, str(idx), color='white', fontsize=12,
                ha='center', weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=0.2, boxstyle='circle'))

    all_x = [p.x for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    all_y = [p.y for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
        ax.set_ylim(min(all_y) - 20, max(all_y) + 20)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Configuration for Scenario: {scenario.scenario_id}", color='white')
    
    config_image_path = os.path.join(output_dir, f"{scenario.scenario_id}_config.png")
    plt.savefig(config_image_path, dpi=150, facecolor='black')
    plt.close(fig)
    
    print(f"\n✅ Created configuration map: {config_image_path}")
    return approaches

def main():
    """
    Main interactive loop to find and define signal phases for scenarios.
    """
    print("--- Interactive Signal Phase Configurator ---")
    
    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))
    if not tfrecord_files:
        print(f"Error: No .tfrecord files found in '{TFRECORD_DIR}'")
        return

    # Load existing config if it exists
    config_data = {}
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            config_data = json.load(f)
            print(f"Loaded existing configurations from '{CONFIG_FILENAME}'")

    for file_path in tfrecord_files:
        print(f"\n>>> Reading file: {os.path.basename(file_path)}")
        dataset = tf.data.TFRecordDataset(file_path)
        
        for record in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())
            
            # --- NEW: Interactive Choice ---
            if scenario.scenario_id in config_data:
                print(f"--- Scenario {scenario.scenario_id} is already configured. Skipping.")
                continue

            try:
                choice = input(f"--- Found Scenario: {scenario.scenario_id}. Configure it? [y/n/q]: ").lower()
            except KeyboardInterrupt:
                print("\nQuitting.")
                return

            if choice == 'q':
                print("Quitting.")
                return
            elif choice == 'n':
                print("Skipping.")
                continue
            elif choice == 'y':
                output_dir = os.path.dirname(file_path) or '.'
                approaches = generate_config_frame(scenario, output_dir)
                num_approaches = len(approaches)

                print(f"\nIdentified {num_approaches} possible approaches (traffic movements).")
                print("Please view the generated map and group the approach numbers into compatible phases.")
                
                phases = []
                phase_num = 1
                while True:
                    try:
                        user_input = input(f"Enter compatible approach numbers for Phase {phase_num}, separated by commas (e.g., 0, 5). Type 'done' to finish: ")
                        if user_input.lower() == 'done':
                            if not phases: print("No phases defined for this scenario.")
                            break
                        
                        phase_approaches = [int(x.strip()) for x in user_input.split(',')]
                        if any(p >= num_approaches or p < 0 for p in phase_approaches):
                            print(f"Error: Please enter numbers between 0 and {num_approaches - 1}.")
                            continue

                        phases.append(phase_approaches)
                        print(f"  > Phase {phase_num} defined as: {phase_approaches}")
                        phase_num += 1
                    except ValueError:
                        print("Invalid input. Please enter numbers separated by commas.")
                    except KeyboardInterrupt:
                        print("\nConfiguration cancelled for this scenario.")
                        phases = [] # Discard partial configuration
                        break
                
                if phases:
                    config_data[scenario.scenario_id] = phases
                    with open(CONFIG_FILENAME, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    print(f"✅ Configuration saved for scenario {scenario.scenario_id} in '{CONFIG_FILENAME}'")
            else:
                print("Invalid choice. Skipping scenario.")

    print("\n--- All files processed. ---")

if __name__ == '__main__':
    main()