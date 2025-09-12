# waymo_queue_detector.py

import os
import glob
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, map_pb2
import numpy as np
from tqdm import tqdm
import warnings
from scipy.spatial import distance
from collections import defaultdict

# Suppress TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
# Directory containing your Waymo Motion Dataset .tfrecord files
# IMPORTANT: Update this path to where your data is stored.
TFRECORD_DIR = '../waymo_movie_file'

# --- Algorithm Parameters ---
# Speed threshold to consider a vehicle "queued". 5 km/h is approx 1.4 m/s.
SPEED_THRESHOLD_MS = 1.4  # in meters/second

LANE_ASSIGNMENT_THRESHOLD = 5.0   # meters, max distance from lane center



def get_slow_vehicles(scenario, timestamp_index):
    """
    Identifies all vehicles moving slower than a threshold at a specific timestamp.
    """
    slow_vehicles = []
    for track in scenario.tracks:
        if track.object_type != track.ObjectType.TYPE_VEHICLE:
            continue
        if is_parked(track):  # <-- NEW check
            continue
        state = track.states[timestamp_index]
        if not state.valid:
            continue
        speed = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
        if speed < SPEED_THRESHOLD_MS:
            slow_vehicles.append({
                'id': track.id,
                'x': state.center_x,
                'y': state.center_y,
                'length': state.length,
                'width': state.width
            })
    return slow_vehicles

def is_parked(track):
    # A vehicle is parked if it barely moves across all valid states
    positions = [(s.center_x, s.center_y) for s in track.states if s.valid]
    if len(positions) < 2:
        return True
    total_disp = np.linalg.norm(np.array(positions[-1]) - np.array(positions[0]))
    return total_disp < 2.0  # meters (tunable threshold)


def assign_vehicles_to_lanes(vehicles, map_features):
    """
    Assigns each vehicle to the nearest lane centerline from the map data.
    Rejects vehicles too far from any lane (likely parked).
    """
    lane_features = [f for f in map_features if f.WhichOneof('feature_data') == 'lane']
    
    for vehicle in vehicles:
        vehicle_pos = np.array([vehicle['x'], vehicle['y']])
        min_dist = float('inf')
        closest_lane_id = -1

        for lane in lane_features:
            lane_points = np.array([[p.x, p.y] for p in lane.lane.polyline])
            if lane_points.shape[0] == 0:
                continue
            
            dists = distance.cdist(vehicle_pos[np.newaxis, :], lane_points)
            current_min_dist = np.min(dists)
            
            if current_min_dist < min_dist:
                min_dist = current_min_dist
                closest_lane_id = lane.id
        
        # Reject if too far from lane centerline
        if min_dist > LANE_ASSIGNMENT_THRESHOLD:
            vehicle['lane_id'] = -1
        else:
            vehicle['lane_id'] = closest_lane_id


def group_vehicles_by_lane(vehicles_with_lanes):
    """
    Groups vehicles into queues based on their assigned lane ID.
    No minimum vehicle requirement.
    """
    queues = {}
    for vehicle in vehicles_with_lanes:
        lane_id = vehicle.get('lane_id')
        if lane_id is None or lane_id == -1:
            continue
        if lane_id not in queues:
            queues[lane_id] = []
        queues[lane_id].append(vehicle)
    
    return queues



def analyze_queue_state(queue_vehicles, lane_feature):
    """
    Analyzes a single queue to get its state (size, length, density).
    """
    queue_size = len(queue_vehicles)
    
    stop_line_pos = None
    if lane_feature.lane.polyline:
         stop_line_pos = np.array([lane_feature.lane.polyline[0].x, lane_feature.lane.polyline[0].y])
    
    if stop_line_pos is None:
        return None

    max_dist = 0
    rearmost_vehicle = None
    for vehicle in queue_vehicles:
        vehicle_pos = np.array([vehicle['x'], vehicle['y']])
        dist = np.linalg.norm(vehicle_pos - stop_line_pos)
        if dist > max_dist:
            max_dist = dist
            rearmost_vehicle = vehicle
    
    if rearmost_vehicle:
        queue_length = max_dist + (rearmost_vehicle['length'] / 2)
        density = (queue_size / queue_length) * 1000 if queue_length > 0 else 0
        vehicle_ids = [v['id'] for v in queue_vehicles]
        
        return {
            "size": queue_size,
            "length": queue_length,
            "density": density,
            "vehicle_ids": vehicle_ids
        }
    return None

def build_adjacency_map(map_features):
    """
    Creates a map of which lanes are adjacent to each other.
    """
    adjacency_map = defaultdict(list)
    lane_features = [f for f in map_features if f.WhichOneof('feature_data') == 'lane']
    for lane in lane_features:
        # Waymo map data directly provides IDs of left and right neighbors
        for neighbor in lane.lane.left_neighbors:
            adjacency_map[lane.id].append(neighbor.feature_id)
        for neighbor in lane.lane.right_neighbors:
            adjacency_map[lane.id].append(neighbor.feature_id)
    return adjacency_map


def process_scenario(scenario):
    """
    Processes a single scenario and prints a summary at the end.
    """
    print(f"\n{'='*60}")
    print(f"Processing Scenario ID: {scenario.scenario_id}")
    
    map_counts = defaultdict(int)
    for feature in scenario.map_features:
        map_counts[feature.WhichOneof('feature_data')] += 1
    
    print("Map Feature Breakdown:")
    for feature_type, count in map_counts.items():
        print(f"  - {feature_type.capitalize()}s: {count}")
    print(f"{'='*60}")

    scenario_queues = defaultdict(list)
    map_feature_dict = {f.id: f for f in scenario.map_features}
    
    # NEW: Build the lane adjacency map once per scenario
    adjacency_map = build_adjacency_map(scenario.map_features)

    for i, timestamp in enumerate(scenario.timestamps_seconds):
        slow_vehicles = get_slow_vehicles(scenario, i)
        if not slow_vehicles:
            continue
            
        assign_vehicles_to_lanes(slow_vehicles, scenario.map_features)
        queues_by_lane = group_vehicles_by_lane(slow_vehicles)
        
        for lane_id, vehicles in queues_by_lane.items():
            lane_feature = map_feature_dict.get(lane_id)
            if lane_feature:
                queue_state = analyze_queue_state(vehicles, lane_feature)
                if queue_state:
                    scenario_queues[lane_id].append(queue_state)

    print("\n--- Scenario Summary ---")
    if not scenario_queues:
        print("No traffic queues were detected in this scenario.")
        return

    # --- Individual Lane Reports ---
    print("\n[Individual Lane Queues]")
    peak_states = {}
    for lane_id, queue_history in scenario_queues.items():
        if not queue_history: continue
        peak_state = max(queue_history, key=lambda x: x['length'])
        peak_states[lane_id] = peak_state
        
        print(f"\nLane ID: {lane_id}")
        print(f"  - State of Queue (at peak):")
        print(f"    - Vehicle IDs in Queue: {peak_state['vehicle_ids']}")
        print(f"    - Queue Length: {peak_state['length']:.1f} meters")
        print(f"  - Estimated Density: {peak_state['density']:.1f} veh/km")

    # --- NEW: Adjacent Lane Cluster Reports ---
    print("\n[Adjacent Lane Congestion Clusters]")
    processed_lanes = set()
    clusters_found = 0
    for lane_id in scenario_queues:
        if lane_id in processed_lanes:
            continue
        
        cluster = {lane_id}
        # Use a queue to find all connected neighbors with queues
        lanes_to_check = [lane_id]
        while lanes_to_check:
            current_lane = lanes_to_check.pop(0)
            for neighbor_id in adjacency_map.get(current_lane, []):
                if neighbor_id in scenario_queues and neighbor_id not in cluster:
                    cluster.add(neighbor_id)
                    lanes_to_check.append(neighbor_id)
        
        processed_lanes.update(cluster)
        
        if len(cluster) > 1:
            clusters_found += 1
            total_vehicles = sum(peak_states[cid]['size'] for cid in cluster)
            total_length = sum(peak_states[cid]['length'] for cid in cluster)
            combined_density = (total_vehicles / total_length) * 1000 if total_length > 0 else 0
            
            print(f"\nCluster #{clusters_found}:")
            print(f"  - Congested Adjacent Lanes: {sorted(list(cluster))}")
            print(f"  - Total Vehicles in Cluster: {total_vehicles}")
            print(f"  - Combined Estimated Density: {combined_density:.1f} veh/km")

    if clusters_found == 0:
        print("No congestion spanning multiple adjacent lanes was detected.")


def main():
    """
    Main function to find and process all .tfrecord files in the specified directory.
    """
    print("--- Waymo Motion Dataset Queue Detector (with Lane ID) ---")
    
    if not os.path.isdir(TFRECORD_DIR):
        print(f"\nError: Input directory not found at '{os.path.abspath(TFRECORD_DIR)}'")
        print("Please update the TFRECORD_DIR variable in the script to the correct path.")
        return

    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))

    if not tfrecord_files:
        print(f"No .tfrecord files found in '{os.path.abspath(TFRECORD_DIR)}'.")
        return
    
    print(f"Found {len(tfrecord_files)} file(s). Starting processing...")

    for file_path in tqdm(tfrecord_files, desc="Overall Progress", unit="file"):
        print(f"\n\n>>> Processing file: {os.path.basename(file_path)} <<<")
        dataset = tf.data.TFRecordDataset(file_path)
        for record in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())
            if scenario.scenario_id != 'c692808f8d63a7ec':
                continue
            process_scenario(scenario)
    
    print("\n--- Processing Complete ---")

if __name__ == '__main__':
    main()
