import os
import glob
import csv
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from tqdm import tqdm

# ---------------- Configuration ----------------
TFRECORD_DIR = "/mnt/c/Users/durga/OneDrive/Desktop/phase skipping/data/"  # Update to your TFRecord folder
OUTPUT_DIR = "/mnt/c/Users/durga/OneDrive/Desktop/phase skipping/output/"  # Directory to store the CSV
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'phase_skipping_output.csv')
MAX_SCENARIOS = 60  # Limit scenarios to process

def is_phase_skippable(lane_state):
    enum_map = lane_state.DESCRIPTOR.fields_by_name["state"].enum_type.values_by_name
    GO = enum_map.get('LANE_STATE_GO', None)
    ARROW_GO = enum_map.get('LANE_STATE_ARROW_GO', None)
    go_val = GO.number if GO else 3
    arrow_go_val = ARROW_GO.number if ARROW_GO else 4
    return not (lane_state.state == go_val or lane_state.state == arrow_go_val)

def process_scenario(scenario):
    rows = []
    scenario_id = scenario.scenario_id

    if not scenario.dynamic_map_states:
        return rows

    for ts_index, dynamic_map_state in enumerate(scenario.dynamic_map_states):
        try:
            timestamp = scenario.timestamps_seconds[ts_index]
        except IndexError:
            continue

        lane_states = getattr(dynamic_map_state, "lane_states", [])
        if not lane_states:
            continue

        for lane_state in lane_states:
            try:
                lane_id = getattr(lane_state, "lane", "Unknown")
                skippable = is_phase_skippable(lane_state)
                rows.append([scenario_id, timestamp, lane_id, skippable])
            except Exception:
                continue
    return rows

def main():
    print("--- Waymo Phase Skipping Detector ---")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(TFRECORD_DIR):
        print(f"Error: Directory not found: {TFRECORD_DIR}")
        return

    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))
    if not tfrecord_files:
        print(f"No TFRecord files found in {TFRECORD_DIR}")
        return

    print(f"Found {len(tfrecord_files)} TFRecord files. Starting processing...")

    all_rows = []
    scenarios_processed = 0

    for file_path in tqdm(tfrecord_files, desc="Processing TFRecords"):
        dataset = tf.data.TFRecordDataset(file_path)
        for record in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())

            scenario_rows = process_scenario(scenario)
            if scenario_rows:
                all_rows.extend(scenario_rows)

            scenarios_processed += 1
            if scenarios_processed >= MAX_SCENARIOS:
                break
        if scenarios_processed >= MAX_SCENARIOS:
            break

    if not all_rows:
        print("No scenario rows processed. CSV not created.")
        return

    # Save to CSV inside the output directory
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario_id', 'timestamp', 'lane_id', 'skippable'])
        writer.writerows(all_rows)

    print(f"CSV saved: {OUTPUT_CSV}")
    print(f"Total scenarios processed: {scenarios_processed}")

if __name__ == "__main__":
    main()
