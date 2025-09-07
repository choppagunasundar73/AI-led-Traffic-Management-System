import os
import glob
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from waymo_open_dataset.protos import scenario_pb2
import csv

# -----------------------------
# Turn classification function (angle-based)
# -----------------------------
def classify_turn(vehicle_xy):
    if len(vehicle_xy) < 2:
        return "Unclassified"

    x_start, y_start = vehicle_xy[0]
    x_end, y_end = vehicle_xy[-1]

    dx = x_end - x_start
    dy = y_end - y_start

    if dx == 0 and dy == 0:
        return "Unclassified"

    # Angle in degrees
    angle = math.degrees(math.atan2(dy, dx))

    # Normalize angle to [-180, 180]
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360

    # Classification thresholds
    if -15 <= angle <= 15:
        return "Straight"
    elif angle > 15:
        return "LeftTurn"
    elif angle < -15:
        return "RightTurn"
    else:
        return "Unclassified"

# -----------------------------
# Paths and TFRecord loading
# -----------------------------
data_dir = "data"
all_files = glob.glob(os.path.join(data_dir, "training*.tfrecord*"))

if not all_files:
    raise FileNotFoundError("❌ No TFRecord files found in data/. Please add dataset files.")

print(f"✅ Found {len(all_files)} TFRecord files in 'data/'.")

# Function to check if a TFRecord file is readable
def is_file_ok(filename):
    try:
        for _ in tf.data.TFRecordDataset(filename, compression_type=None):
            pass
        return True
    except Exception:
        print(f"⚠️ Skipping corrupted or incomplete file: {filename}")
        return False

# Keep only valid files
tfrecord_files = [f for f in all_files if is_file_ok(f)]
print(f"✅ Using {len(tfrecord_files)} valid TFRecord files.")

# -----------------------------
# Output directories
# -----------------------------
base_plot_dir = "plots_all_vehicles"
os.makedirs(base_plot_dir, exist_ok=True)
for folder in ["Straight", "LeftTurn", "RightTurn", "Unclassified"]:
    os.makedirs(os.path.join(base_plot_dir, folder), exist_ok=True)

# CSV to store classifications
csv_path = os.path.join(base_plot_dir, "classified_turns_all_vehicles.csv")
processed_pairs = set()  # (scenario_id, vehicle_index)

# Load processed pairs if CSV exists
if os.path.exists(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and len(row) >= 2:
                processed_pairs.add((row[0], int(row[1])))

print(f"🔄 Resuming from last run. Already processed {len(processed_pairs)} (scenario, vehicle) pairs.")

# -----------------------------
# Extra filters for PNG saving
# -----------------------------
scenario_limit = 50      # max scenarios
scenario_count = 0

# -----------------------------
# Open CSV for appending new entries
# -----------------------------
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if os.path.getsize(csv_path) == 0:  # empty file → new header
        writer.writerow(["ScenarioID", "VehicleIndex", "Classification"])

    # -----------------------------
    # Process each TFRecord file
    # -----------------------------
    for filename in tfrecord_files:
        print(f"\n📂 Processing file: {filename}")

        try:
            dataset = tf.data.TFRecordDataset(filename, compression_type=None)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
            continue

        for i, data in enumerate(dataset):
            if scenario_count >= scenario_limit:
                break  # stop after 50 scenarios

            try:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(data.numpy())
            except Exception as e:
                print(f"⚠️ Skipping scenario {i} in {filename} due to parse error: {e}")
                continue

            scenario_id = scenario.scenario_id
            scenario_count += 1

            print(f"\n--- Scenario {i} ---")
            print(f"Scenario ID: {scenario_id}")
            print(f"Number of tracks: {len(scenario.tracks)}")
            print(f"Number of map features: {len(scenario.map_features)}")

            # -----------------------------
            # Process all vehicles in the scenario
            # -----------------------------
            for v_idx, track in enumerate(scenario.tracks):

                # Skip if already processed
                if (scenario_id, v_idx) in processed_pairs:
                    continue
                processed_pairs.add((scenario_id, v_idx))

                # Extract vehicle trajectory
                vehicle_xy = [(s.center_x, s.center_y) for s in track.states if s.valid]

                # Classify using angle-based method
                classification = classify_turn(vehicle_xy)

                # -----------------------------
                # Apply filters before saving PNG
                # -----------------------------
                save_plot = False

                if classification in ["Straight", "LeftTurn", "RightTurn"]:
                    if len(vehicle_xy) >= 5:  # only good clear tracks
                        save_plot = True
                elif classification == "Unclassified":
                    save_plot = True  # save all unclassified

                if save_plot:
                    save_dir = os.path.join(base_plot_dir, classification)
                    save_path = os.path.join(save_dir, f"{scenario_id}_vehicle{v_idx}.png")

                    plt.figure(figsize=(8, 8))
                    for mf in scenario.map_features:
                        if mf.HasField("lane"):
                            xs, ys = zip(*[(p.x, p.y) for p in mf.lane.polyline])
                            plt.plot(xs, ys, "gray", linewidth=1)

                    if vehicle_xy:
                        xs, ys = zip(*vehicle_xy)
                        plt.plot(xs, ys, "r-", linewidth=2, label=f"Vehicle {v_idx}")

                    plt.title(f"Scenario {i} Vehicle {v_idx} ({classification})")
                    plt.legend()
                    plt.axis("equal")
                    plt.savefig(save_path)
                    plt.close()

                    print(f"🚗 Vehicle {v_idx} classified as: {classification}")
                    print(f"✅ Saved plot to {save_path}")
                else:
                    print(f"🚗 Vehicle {v_idx} classified as: {classification} (PNG skipped)")

                # Write classification to CSV (always recorded)
                writer.writerow([scenario_id, v_idx, classification])
                f.flush()

        if scenario_count >= scenario_limit:
            break

print(f"\n📂 All classifications saved to {csv_path}")
print(f"✅ Total scenarios processed: {scenario_count}")
