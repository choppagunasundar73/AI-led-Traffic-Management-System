import os
import glob
import json
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import numpy as np
from tqdm import tqdm
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from moviepy.editor import ImageSequenceClip
from scipy.spatial import distance

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
TFRECORD_DIR = '../waymo_movie_file'
OUTPUT_DIR = '../signal_emulation_videos_final'
CONFIG_FILENAME = 'scenario_phases.json'

# --- Actuated Signal Control Parameters ---
MIN_GREEN = 7.0
MAX_GREEN = 60.0
PASSAGE_TIME = 4.0
ALL_RED_INTERVAL = 2.0
VEHICLE_WAITING_DIST = 15.0
STOP_LINE_PROXIMITY = 10.0

# --- Visualization Parameters ---
LANE_COLOR_GREEN = 'lime'
LANE_COLOR_RED = '#8B0000'
STOP_LINE_GREEN = 'lime'
STOP_LINE_RED = 'red'
VEHICLE_COLOR_DEFAULT = 'royalblue'
VEHICLE_COLOR_CROSSING = 'yellow'
MAP_COLOR = 'gray'
FIG_DPI = 120

# --- Core Logic Functions (Optimized) ---

def group_lanes_into_approaches(map_features):
    lanes = [f for f in map_features if f.WhichOneof('feature_data') == 'lane' and len(f.lane.polyline) > 1]
    approaches, processed_lane_ids = [], set()
    for i, lane1 in enumerate(lanes):
        if lane1.id in processed_lane_ids: continue
        current_approach = {'lane_ids': {lane1.id}, 'stop_line_pos': np.array([lane1.lane.polyline[0].x, lane1.lane.polyline[0].y])}
        processed_lane_ids.add(lane1.id)
        for j in range(i + 1, len(lanes)):
            lane2 = lanes[j]
            if lane2.id in processed_lane_ids: continue
            stop_line2_pos = np.array([lane2.lane.polyline[0].x, lane2.lane.polyline[0].y])
            if np.linalg.norm(current_approach['stop_line_pos'] - stop_line2_pos) < STOP_LINE_PROXIMITY:
                current_approach['lane_ids'].add(lane2.id); processed_lane_ids.add(lane2.id)
        approaches.append(current_approach)
    return approaches

def get_vehicle_lane_assignment(state, lanes_with_points):
    min_dist, closest_lane_id = float('inf'), None
    vehicle_pos = np.array([state.center_x, state.center_y])
    for lane_id, points in lanes_with_points.items():
        dist = np.min(distance.cdist(vehicle_pos[np.newaxis, :], points))
        if dist < min_dist:
            min_dist, closest_lane_id = dist, lane_id
    return closest_lane_id

def precompute_vehicle_lanes(scenario, map_features):
    vehicle_lanes = {}
    lanes_with_points = {f.id: np.array([[p.x, p.y] for p in f.lane.polyline]) for f in map_features if f.WhichOneof('feature_data') == 'lane' and len(f.lane.polyline) > 0}
    for track in scenario.tracks:
        if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
        for i, state in enumerate(track.states):
            if state.valid:
                vehicle_lanes[(track.id, i)] = get_vehicle_lane_assignment(state, lanes_with_points)
    return vehicle_lanes

def precompute_crossing_times(scenario, approaches, vehicle_lanes, map_features):
    crossing_events = defaultdict(list)
    dt = np.mean(np.diff(scenario.timestamps_seconds))
    for track in scenario.tracks:
        if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
        for i in range(1, len(track.states)):
            prev_state, curr_state = track.states[i-1], track.states[i]
            if not (prev_state.valid and curr_state.valid): continue
            for approach_idx, approach in enumerate(approaches):
                lane_id = vehicle_lanes.get((track.id, i))
                if lane_id not in approach['lane_ids']: continue
                lane_dir = get_lane_direction(lane_id, map_features)
                if lane_dir is None: continue
                stop_line_pos = approach['stop_line_pos']
                vec_prev = np.array([prev_state.center_x, prev_state.center_y]) - stop_line_pos
                vec_curr = np.array([curr_state.center_x, curr_state.center_y]) - stop_line_pos
                if np.dot(vec_prev, lane_dir) < 0 and np.dot(vec_curr, lane_dir) >= 0:
                    dist_to_line = abs(np.dot(vec_prev, lane_dir))
                    speed_towards_line = abs(np.dot(np.array([prev_state.velocity_x, prev_state.velocity_y]), lane_dir))
                    time_to_cross = dist_to_line / speed_towards_line if speed_towards_line > 0.1 else dt / 2
                    crossing_time = scenario.timestamps_seconds[i-1] + time_to_cross
                    crossing_events[approach_idx].append({'time': crossing_time, 'vehicle_id': track.id})
                    break
    for approach_idx in crossing_events:
        crossing_events[approach_idx].sort(key=lambda x: x['time'])
    return crossing_events

def get_lane_direction(lane_id, map_features):
    for f in map_features:
        if f.id == lane_id and len(f.lane.polyline) > 1:
            p0 = np.array([f.lane.polyline[0].x, f.lane.polyline[0].y])
            p1 = np.array([f.lane.polyline[1].x, f.lane.polyline[1].y])
            direction = p1 - p0
            return direction / np.linalg.norm(direction)
    return None

def check_for_demand(scenario, timestamp_idx, approach, vehicle_lanes, map_features):
    for track in scenario.tracks:
        state = track.states[timestamp_idx]
        if not state.valid or track.object_type != track.ObjectType.TYPE_VEHICLE: continue
        lane_id = vehicle_lanes.get((track.id, timestamp_idx))
        if lane_id in approach['lane_ids']:
            speed = np.linalg.norm([state.velocity_x, state.velocity_y])
            dist_to_stop = np.linalg.norm(np.array([state.center_x, state.center_y]) - approach['stop_line_pos'])
            if speed < 1.0 and dist_to_stop < VEHICLE_WAITING_DIST:
                lane_dir = get_lane_direction(lane_id, map_features)
                if lane_dir is not None and np.dot(np.array([state.center_x, state.center_y]) - approach['stop_line_pos'], lane_dir) < 0:
                    return True
    return False

# --- Main Animation Function ---

def create_animation_for_scenario(scenario, file_basename, scenario_index, phases_config):
    phases = phases_config.get(scenario.scenario_id)
    if not phases:
        print(f"  No phase configuration found for scenario {scenario.scenario_id}. Skipping.")
        return

    video_filename = f"{file_basename}_scenario_{scenario_index:03d}_{scenario.scenario_id}_PhasedSignalLogic.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    frame_dir = os.path.join(OUTPUT_DIR, f"temp_frames_{scenario.scenario_id}")
    os.makedirs(frame_dir, exist_ok=True)

    print(f"\n--- Emulating Phased Signal for Scenario {scenario.scenario_id} ---")

    approaches = group_lanes_into_approaches(scenario.map_features)
    vehicle_lanes = precompute_vehicle_lanes(scenario, scenario.map_features)
    crossing_events = precompute_crossing_times(scenario, approaches, vehicle_lanes, scenario.map_features)
    print("  Preprocessing complete. Generating frames...")

    map_feature_dict = {f.id: f for f in scenario.map_features}
    phase_states = [{'state': 'RED', 'green_timer': 0.0, 'last_crossing': -1, 'termination_reason': ''} for _ in phases]
    last_served_phase = -1
    all_red_timer = 0.0

    all_x = [p.x for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    all_y = [p.y for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    x_min, x_max, y_min, y_max = min(all_x)-20, max(all_x)+20, min(all_y)-20, max(all_y)+20
    frame_files = []
    
    for i in tqdm(range(len(scenario.timestamps_seconds)), desc="  Generating Frames"):
        current_time = scenario.timestamps_seconds[i]
        dt = (scenario.timestamps_seconds[i] - scenario.timestamps_seconds[i-1]) if i > 0 else 0
        fig, ax = plt.subplots(figsize=(12, 12)); ax.set_facecolor('black')

        if all_red_timer > 0: all_red_timer -= dt
        
        current_green_phase = -1
        for idx, state in enumerate(phase_states):
            if state['state'] == 'GREEN': current_green_phase = idx; break
        
        # Synchronization Step: Runs on every frame to catch vehicles crossing red lights.
        sync_event_occurred = False
        for approach_idx in range(len(approaches)):
            target_phase_idx = -1
            for p_idx, phase in enumerate(phases):
                if approach_idx in phase: target_phase_idx = p_idx; break
            
            if target_phase_idx != -1 and phase_states[target_phase_idx]['state'] == 'RED':
                if any(abs(c['time'] - current_time) < dt*1.5 for c in crossing_events[approach_idx]):
                    if current_green_phase != -1:
                        phase_states[current_green_phase]['state'] = 'RED'; phase_states[current_green_phase]['termination_reason'] = 'PRE-EMPT'
                        last_served_phase = current_green_phase
                    
                    state = phase_states[target_phase_idx]
                    state['state'] = 'GREEN'; state['green_timer'] = 0.0; state['last_crossing'] = current_time; state['termination_reason'] = ''
                    current_green_phase = target_phase_idx
                    all_red_timer = 0
                    sync_event_occurred = True
                    break
        
        if sync_event_occurred:
            pass # Skip other logic for this frame if a sync happened
        elif current_green_phase != -1: # Termination Check
            state = phase_states[current_green_phase]
            state['green_timer'] += dt
            most_recent_crossing = state['last_crossing']
            for approach_idx in phases[current_green_phase]:
                crossings = [c['time'] for c in crossing_events[approach_idx] if c['time'] <= current_time]
                if crossings: most_recent_crossing = max(most_recent_crossing, max(crossings))
            state['last_crossing'] = most_recent_crossing
            gap = current_time - state['last_crossing']
            terminated = False
            if state['green_timer'] > MIN_GREEN:
                if state['green_timer'] >= MAX_GREEN: state['termination_reason'] = 'MAX-OUT'; terminated = True
                elif gap > PASSAGE_TIME: state['termination_reason'] = 'GAP-OUT'; terminated = True
            if terminated:
                state['state'] = 'RED'; last_served_phase = current_green_phase; all_red_timer = ALL_RED_INTERVAL; current_green_phase = -1
        elif current_green_phase == -1 and all_red_timer <= 0: # Activation Check
            for offset in range(len(phases)):
                next_phase_idx = (last_served_phase + 1 + offset) % len(phases)
                phase_has_demand = any(check_for_demand(scenario, i, approaches[app_idx], vehicle_lanes, scenario.map_features) for app_idx in phases[next_phase_idx])
                if phase_has_demand:
                    state = phase_states[next_phase_idx]
                    state['state'] = 'GREEN'; state['green_timer'] = 0.0; state['last_crossing'] = current_time; state['termination_reason'] = ''
                    break
        
        # --- Drawing Logic ---
        green_approaches = set()
        if current_green_phase != -1:
            for app_idx in phases[current_green_phase]: green_approaches.add(app_idx)

        for idx, approach in enumerate(approaches):
            is_green = idx in green_approaches
            lane_color, stop_line_color = (LANE_COLOR_GREEN, STOP_LINE_GREEN) if is_green else (LANE_COLOR_RED, STOP_LINE_RED)
            for lane_id in approach['lane_ids']:
                lane_feature = map_feature_dict.get(lane_id)
                if lane_feature and hasattr(lane_feature.lane, 'polyline'):
                    points = np.array([[p.x, p.y] for p in lane_feature.lane.polyline])
                    if points.shape[0] > 1: ax.plot(points[:, 0], points[:, 1], color=lane_color, linewidth=is_green*4+1, alpha=0.8, zorder=2)
            stop_line_pos = approach['stop_line_pos']
            ax.text(stop_line_pos[0], stop_line_pos[1] + 5, str(idx), color='white', fontsize=10, ha='center', zorder=5, bbox=dict(facecolor='black', alpha=0.6, pad=0.1, boxstyle='circle'))
            lane_dir = get_lane_direction(list(approach['lane_ids'])[0], scenario.map_features)
            if lane_dir is not None:
                perp_vec = np.array([-lane_dir[1], lane_dir[0]])
                start_pt, end_pt = stop_line_pos - perp_vec * 5, stop_line_pos + perp_vec * 5
                ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color=stop_line_color, linewidth=4, zorder=3)
        
        active_phase_for_text = current_green_phase if current_green_phase != -1 else last_served_phase if all_red_timer > 0 else -1
        if active_phase_for_text != -1:
            state_info = phase_states[active_phase_for_text]
            if state_info['state'] == 'GREEN':
                gap = current_time - state_info['last_crossing']
                plt.figtext(0.1, 0.95, f"Phase {active_phase_for_text} Status: GREEN (Approaches: {phases[active_phase_for_text]})", color='lime', fontsize=12, weight='bold')
                plt.figtext(0.1, 0.92, f"  Green Time: {state_info['green_timer']:.1f}s / {MAX_GREEN:.1f}s", color='white')
                plt.figtext(0.1, 0.89, f"  Phase Gap: {gap:.1f}s / {PASSAGE_TIME:.1f}s", color='cyan' if gap <= PASSAGE_TIME else 'orange')
            elif state_info['termination_reason']:
                plt.figtext(0.1, 0.95, f"Phase {active_phase_for_text} Status: RED", color='red', fontsize=12)
                plt.figtext(0.1, 0.92, f"  Termination: {state_info['termination_reason']}", color='red', weight='bold')

        for track in scenario.tracks:
            if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
            state = track.states[i]
            if not state.valid: continue
            color = VEHICLE_COLOR_DEFAULT
            for approach_idx in crossing_events:
                if any(abs(c['time'] - current_time) < 0.1 and c['vehicle_id'] == track.id for c in crossing_events[approach_idx]):
                    color = VEHICLE_COLOR_CROSSING; break
            rect = patches.Rectangle((state.center_x-state.length/2, state.center_y-state.width/2), state.length, state.width, edgecolor='white', facecolor=color, zorder=4, lw=0.5)
            transform = mtransforms.Affine2D().rotate_around(state.center_x, state.center_y, state.heading) + ax.transData
            rect.set_transform(transform); ax.add_patch(rect)

        ax.set_aspect('equal', adjustable='box'); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f"Phased Actuated Signal Emulation | Time: {current_time:.1f}s", color='white')
        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=FIG_DPI, facecolor='black'); plt.close(fig)
        frame_files.append(frame_path)

    print(f"  Compiling frames into video: {video_filename}...")
    try:
        clip = ImageSequenceClip(frame_files, fps=10)
        clip.write_videofile(video_path, codec='libx264', logger=None)
        print(f"  ✅ Video saved successfully!")
    except Exception as e: print(f"  ❌ Error: {e}")
    finally:
        for f in frame_files:
            try: os.remove(f)
            except OSError: pass
        try: os.rmdir(frame_dir)
        except OSError: pass

def main():
    print("--- Waymo Phased Actuated Signal Reconstructor ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILENAME, 'r') as f:
            phases_config = json.load(f)
        print(f"Loaded phase configurations from '{CONFIG_FILENAME}'")
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILENAME}' not found.")
        print("Please run the 'configure_signal_phases.py' script first.")
        return

    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))
    if not tfrecord_files: print(f"Error: No .tfrecord files found in '{TFRECORD_DIR}'"); return
    
    for file_path in tfrecord_files:
        file_basename = os.path.basename(file_path).split('.tfrecord')[0]
        dataset = tf.data.TFRecordDataset(file_path)
        try:
            for i, record in enumerate(dataset):
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                if scenario.scenario_id in phases_config:
                    create_animation_for_scenario(scenario, file_basename, i + 1, phases_config)
        except Exception as e: print(f"An unexpected error occurred: {e}")
    print("\n--- All configured scenarios processed. ---")

if __name__ == '__main__':
    main()