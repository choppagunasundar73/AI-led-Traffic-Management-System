import os
import glob
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import numpy as np
from tqdm import tqdm
import warnings
from scipy.spatial import distance
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from moviepy.editor import ImageSequenceClip

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
TFRECORD_DIR = '../waymo_movie_file'
OUTPUT_DIR = '../traffic_videos'  # Output videos will be saved here

# --- Algorithm Parameters ---
SPEED_THRESHOLD_MS = 1.4
# NEW: Threshold to consider a vehicle parked if it's too far from a lane.
PARKED_LANE_DISTANCE_THRESHOLD = 2.0 # meters

# --- Visualization Parameters ---
VEHICLE_COLOR_DEFAULT = 'royalblue'
VEHICLE_COLOR_QUEUED = 'red'
LANE_HIGHLIGHT_COLOR = 'yellow'
MAP_COLOR = 'gray'
FIG_DPI = 120

# --- Helper Functions ---

def get_slow_vehicles(scenario, timestamp_index):
    slow_vehicles = []
    for track in scenario.tracks:
        if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
        state = track.states[timestamp_index]
        if not state.valid: continue
        speed = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
        if speed < SPEED_THRESHOLD_MS:
            slow_vehicles.append({'id': track.id, 'x': state.center_x, 'y': state.center_y})
    return slow_vehicles

def filter_out_parked_vehicles(vehicles, map_features):
    """
    NEW FUNCTION: Filters a list of vehicles, removing those too far from any lane.
    """
    on_road_vehicles = []
    lane_features = [f for f in map_features if f.WhichOneof('feature_data') == 'lane']
    
    if not lane_features:
        return vehicles # Return all if no lane data is available

    all_lane_points = []
    for lane in lane_features:
        if hasattr(lane.lane, 'polyline') and len(lane.lane.polyline) > 0:
            all_lane_points.extend([[p.x, p.y] for p in lane.lane.polyline])

    if not all_lane_points:
        return vehicles # Return all if lanes have no points
        
    all_lane_points_np = np.array(all_lane_points)

    for vehicle in vehicles:
        vehicle_pos = np.array([[vehicle['x'], vehicle['y']]])
        dists = distance.cdist(vehicle_pos, all_lane_points_np)
        min_dist_to_any_lane = np.min(dists)
        
        # Only keep vehicles that are close enough to a lane
        if min_dist_to_any_lane <= PARKED_LANE_DISTANCE_THRESHOLD:
            on_road_vehicles.append(vehicle)
            
    return on_road_vehicles

def assign_vehicles_to_lanes(vehicles, map_features):
    lane_features = [f for f in map_features if f.WhichOneof('feature_data') == 'lane']
    for vehicle in vehicles:
        vehicle_pos = np.array([vehicle['x'], vehicle['y']])
        min_dist, closest_lane_id = float('inf'), -1
        for lane in lane_features:
            if not hasattr(lane.lane, 'polyline'): continue
            lane_points = np.array([[p.x, p.y] for p in lane.lane.polyline])
            if lane_points.shape[0] == 0: continue
            dists = distance.cdist(vehicle_pos[np.newaxis, :], lane_points)
            if np.min(dists) < min_dist:
                min_dist, closest_lane_id = np.min(dists), lane.id
        vehicle['lane_id'] = closest_lane_id

def group_vehicles_by_lane(vehicles_with_lanes):
    queues = defaultdict(list)
    for vehicle in vehicles_with_lanes:
        lane_id = vehicle.get('lane_id', -1)
        if lane_id != -1:
            queues[lane_id].append(vehicle)
    return queues

# --- Visualization and Animation ---

def create_animation_for_scenario(scenario, file_basename, scenario_index):
    """
    Generates frames for a scenario, then uses moviepy to create and save a video.
    """
    video_filename = f"{file_basename}_scenario_{scenario_index:03d}_{scenario.scenario_id}.mp4"
    # video_filename = f'{scenario_index:03d}.mp4'
    # video_filename = 'hello.mp4'
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    frame_dir = os.path.join(OUTPUT_DIR, f"temp_frames_{scenario.scenario_id}")
    os.makedirs(frame_dir, exist_ok=True)
    
    print(f"\n--- Processing Scenario {scenario_index} ({scenario.scenario_id}) from {file_basename}.tfrecord ---")

    all_x, all_y = [], []
    for feature in scenario.map_features:
        feature_data = getattr(feature, feature.WhichOneof('feature_data'))
        if hasattr(feature_data, 'polyline'):
            for p in feature_data.polyline:
                all_x.append(p.x)
                all_y.append(p.y)
    
    if not all_x or not all_y:
        print("  Scenario contains no map data to establish boundaries. Skipping.")
        return
        
    x_min, x_max = min(all_x) - 20, max(all_x) + 20
    y_min, y_max = min(all_y) - 20, max(all_y) + 20

    frame_files = []
    num_timestamps = len(scenario.timestamps_seconds)
    for i in tqdm(range(num_timestamps), desc=f"  Generating Frames for Scenario {scenario_index}"):
        fig, ax = plt.subplots(figsize=(12, 12))
        
        for feature in scenario.map_features:
            feature_data = getattr(feature, feature.WhichOneof('feature_data'))
            if hasattr(feature_data, 'polyline'):
                points = np.array([[p.x, p.y] for p in feature_data.polyline])
                if points.shape[0] > 1:
                    ax.plot(points[:, 0], points[:, 1], color=MAP_COLOR, linewidth=0.7)

        # --- MODIFIED QUEUE DETECTION LOGIC ---
        slow_vehicles = get_slow_vehicles(scenario, i)
        # NEW STEP: Filter out parked vehicles before assigning to lanes
        on_road_slow_vehicles = filter_out_parked_vehicles(slow_vehicles, scenario.map_features)
        
        # Subsequent steps now use the filtered list
        assign_vehicles_to_lanes(on_road_slow_vehicles, scenario.map_features)
        queues_by_lane = group_vehicles_by_lane(on_road_slow_vehicles)
        # --- END OF MODIFICATION ---

        map_feature_dict = {f.id: f for f in scenario.map_features}
        for lane_id in queues_by_lane:
            lane_feature = map_feature_dict.get(lane_id)
            if lane_feature and hasattr(lane_feature.lane, 'polyline'):
                points = np.array([[p.x, p.y] for p in lane_feature.lane.polyline])
                if points.shape[0] > 1:
                    ax.plot(points[:, 0], points[:, 1], color=LANE_HIGHLIGHT_COLOR, linewidth=6, alpha=0.5, zorder=2)
        
        queued_vehicle_ids = {v['id'] for queue in queues_by_lane.values() for v in queue}
        for track in scenario.tracks:
            if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
            state = track.states[i]
            if not state.valid: continue
            color = VEHICLE_COLOR_QUEUED if track.id in queued_vehicle_ids else VEHICLE_COLOR_DEFAULT
            rect = patches.Rectangle((state.center_x - state.length / 2, state.center_y - state.width / 2),
                                     state.length, state.width, edgecolor='black', facecolor=color, zorder=3)
            transform = mtransforms.Affine2D().rotate_around(state.center_x, state.center_y, state.heading) + ax.transData
            rect.set_transform(transform)
            ax.add_patch(rect)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Scenario: {scenario.scenario_id} | Time: {scenario.timestamps_seconds[i]:.1f}s")
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")
        
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=FIG_DPI)
        plt.close(fig)
        frame_files.append(frame_path)

    print(f"  Compiling frames into video: {video_filename}...")
    try:
        clip = ImageSequenceClip(frame_files, fps=10)
        clip.write_videofile(video_path, codec='libx264', logger=None)
        print(f"  ✅ Video saved successfully!")
    except Exception as e:
        print(f"  ❌ Error during video compilation: {e}")
    finally:
        print("  Cleaning up temporary frame files...")
        for f in frame_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(frame_dir)
        except OSError:
            pass

def main():
    """Main function to find and process all scenarios in all tfrecord files."""
    print("--- Waymo Queue Visualizer (Batch Processing with Parking Filter) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))
    
    if not tfrecord_files:
        print(f"Error: No .tfrecord files found in '{TFRECORD_DIR}'")
        return
    
    print(f"Found {len(tfrecord_files)} files to process.")
    ids = {'dfbfcbfc1b6f7f7a','c453b2059c68c41c'  , 'f346701fdc8818d1' , 'cb3413b9e69ae5ab' , 'cbcf4099dfd4f9fb'}
    for file_path in tfrecord_files:
        file_basename = os.path.basename(file_path).split('.tfrecord')[0]
        dataset = tf.data.TFRecordDataset(file_path)
        
        try:
            for i, record in enumerate(dataset):
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                if scenario.scenario_id in ids:
                    continue
                create_animation_for_scenario(scenario, file_basename=file_basename, scenario_index=i + 1)
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_basename}: {e}")

    print("\n--- All files and scenarios processed. ---")

if __name__ == '__main__':
    main()