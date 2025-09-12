
import os
import glob
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

# Suppress TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Configuration ---
TFRECORD_DIR = '../waymo_movie_file'
OUTPUT_DIR = '../eta_videos'

# --- Algorithm Parameters ---
LANE_ASSIGNMENT_THRESHOLD = 5.0  # meters
MIN_SPEED_FOR_ETA = 0.5   # meters/second

# --- Visualization Parameters ---
VEHICLE_COLOR_DEFAULT = 'royalblue'
VEHICLE_COLOR_WITH_ETA = 'green'
LANE_HIGHLIGHT_COLOR = 'cyan'
STOP_LINE_COLOR = 'red'
MAP_COLOR = 'gray'
FIG_DPI = 120

# --- Core Logic Functions ---

def get_lane_assignment_and_distance(vehicle_pos, vehicle_velocity, map_features):
    """
    Dynamically determines the correct stop line based on vehicle velocity and
    calculates the distance along the lane to it.
    Returns: (lane_id, distance_to_stop, stop_line_position)
    """
    lane_features = [f for f in map_features if f.WhichOneof('feature_data') == 'lane']
    best_lane_id, min_perp_dist, final_dist, final_stop_line = None, float('inf'), None, None

    for lane in lane_features:
        polyline = np.array([[p.x, p.y] for p in lane.lane.polyline])
        if polyline.shape[0] < 2:
            continue

        min_dist_sq, closest_seg_idx, proj_point = float('inf'), -1, None
        for i in range(len(polyline) - 1):
            p1, p2 = polyline[i], polyline[i + 1]
            line_vec, pnt_vec = p2 - p1, vehicle_pos - p1
            line_len_sq = np.dot(line_vec, line_vec)
            t = 0 if line_len_sq == 0 else max(0, min(1, np.dot(pnt_vec, line_vec) / line_len_sq))
            proj = p1 + t * line_vec
            dist_sq = np.sum((vehicle_pos - proj)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq, closest_seg_idx, proj_point = dist_sq, i, proj
 

        current_perp_dist = np.sqrt(min_dist_sq)
        if current_perp_dist < min_perp_dist:
            min_perp_dist = current_perp_dist
            best_lane_id = lane.id

            if closest_seg_idx != -1:
                # --- NEW: Dynamic Stop Line Logic ---
                seg_p1 = polyline[closest_seg_idx]
                seg_p2 = polyline[closest_seg_idx + 1]
                segment_vec = seg_p2 - seg_p1

                # Use dot product to see if vehicle velocity aligns with polyline order
                dot_product = np.dot(vehicle_velocity, segment_vec)

                if dot_product >= 0: # Vehicle is moving WITH the polyline order
                    stop_line = polyline[-1] # The stop line is at the END
                    dist = np.linalg.norm(seg_p2 - proj_point)
                    for i in range(closest_seg_idx + 1, len(polyline) - 1):
                        dist += np.linalg.norm(polyline[i+1] - polyline[i])
                else: # Vehicle is moving AGAINST the polyline order
                    stop_line = polyline[0] # The stop line is at the START
                    dist = np.linalg.norm(proj_point - seg_p1)
                    for i in range(closest_seg_idx, 0, -1):
                        dist += np.linalg.norm(polyline[i] - polyline[i-1])
                
                final_dist = dist
                final_stop_line = stop_line

    if min_perp_dist <= LANE_ASSIGNMENT_THRESHOLD:
        return best_lane_id, final_dist, final_stop_line
    else:
        return None, None, None

# --- Visualization and Animation ---

def create_animation_for_scenario(scenario, file_basename, scenario_index):
    """Generates frames and compiles them into an MP4 video showing ETA."""
    video_filename = f"{file_basename}_scenario_{scenario_index:03d}_{scenario.scenario_id}_ETA.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    frame_dir = os.path.join(OUTPUT_DIR, f"temp_frames_{scenario.scenario_id}")
    os.makedirs(frame_dir, exist_ok=True)

    print(f"\n--- Processing Scenario {scenario_index} ({scenario.scenario_id}) for ETA Visualization ---")

    all_x = [p.x for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    all_y = [p.y for f in scenario.map_features if hasattr(getattr(f, f.WhichOneof('feature_data')), 'polyline') for p in getattr(f, f.WhichOneof('feature_data')).polyline]
    if not all_x or not all_y:
        print("  Scenario contains no map data. Skipping.")
        return

    x_min, x_max, y_min, y_max = min(all_x) - 20, max(all_x) + 20, min(all_y) - 20, max(all_y) + 20
    frame_files, map_feature_dict = [], {f.id: f for f in scenario.map_features}

    for i in tqdm(range(len(scenario.timestamps_seconds)), desc=f"  Generating Frames"):
        fig, ax = plt.subplots(figsize=(12, 12)); ax.set_facecolor('black')

        for feature in scenario.map_features:
            feature_data = getattr(feature, feature.WhichOneof('feature_data'))
            if hasattr(feature_data, 'polyline') and len(feature_data.polyline) > 1:
                ax.plot(*np.array([[p.x, p.y] for p in feature_data.polyline]).T, color=MAP_COLOR, linewidth=0.7)

        for track in scenario.tracks:
            if track.object_type != track.ObjectType.TYPE_VEHICLE: continue
            state = track.states[i]
            if not state.valid: continue

            eta, assigned_lane_id, stop_line_pos = None, None, None
            speed = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
            if speed > MIN_SPEED_FOR_ETA:
                vehicle_pos = np.array([state.center_x, state.center_y])
                vehicle_velocity = np.array([state.velocity_x, state.velocity_y])
                lane_id, distance, stop_line = get_lane_assignment_and_distance(vehicle_pos, vehicle_velocity, scenario.map_features)
                if lane_id is not None and distance is not None and distance > 0:
                    eta = distance / speed
                    eta = np.clip(eta, 0.1, 8.0) 
                    assigned_lane_id, stop_line_pos = lane_id, stop_line

            color = VEHICLE_COLOR_WITH_ETA if eta is not None else VEHICLE_COLOR_DEFAULT
            rect = patches.Rectangle((state.center_x - state.length/2, state.center_y - state.width/2),
                                     state.length, state.width, edgecolor='white', facecolor=color, zorder=4, lw=0.5)
            transform = mtransforms.Affine2D().rotate_around(state.center_x, state.center_y, state.heading) + ax.transData
            rect.set_transform(transform); ax.add_patch(rect)

            if eta is not None and assigned_lane_id is not None:
                lane_feature = map_feature_dict.get(assigned_lane_id)
                if lane_feature and hasattr(lane_feature.lane, 'polyline'):
                    points = np.array([[p.x, p.y] for p in lane_feature.lane.polyline])
                    if points.shape[0] > 1:
                        ax.plot(points[:, 0], points[:, 1], color=LANE_HIGHLIGHT_COLOR, linewidth=2, alpha=0.8, zorder=2)
                        
                        # Draw stop line marker at the dynamically determined position
                        p0, p1 = (stop_line, points[1]) if np.array_equal(stop_line, points[0]) else (stop_line, points[-2])
                        direction_vector = p1 - p0
                        norm = np.linalg.norm(direction_vector)
                        if norm > 0:
                            perp_vector = np.array([-direction_vector[1], direction_vector[0]]) / norm
                            start_pt, end_pt = p0 - perp_vector * 2.0, p0 + perp_vector * 2.0
                            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]],
                                    color=STOP_LINE_COLOR, linewidth=3, zorder=3, solid_capstyle='round')

                ax.text(state.center_x, state.center_y + 4, f"ETA: {eta:.1f}s", color='white',
                        fontsize=8, ha='center', zorder=5,
                        bbox=dict(facecolor='black', alpha=0.6, pad=0.5, boxstyle='round,pad=0.3'))

        ax.set_aspect('equal', adjustable='box'); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f"ETA to Stop Line | Scenario: {scenario.scenario_id} | Time: {scenario.timestamps_seconds[i]:.1f}s", color='white')
        ax.set_xlabel("X coordinate (m)", color='white'); ax.set_ylabel("Y coordinate (m)", color='white')
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=FIG_DPI, facecolor='black'); plt.close(fig)
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
            try: os.remove(f)
            except OSError: pass
        try: os.rmdir(frame_dir)
        except OSError: pass

def main():
    print("--- Waymo Dynamic ETA Visualizer ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tfrecord_files = glob.glob(os.path.join(TFRECORD_DIR, '*.tfrecord*'))
    if not tfrecord_files:
        print(f"Error: No .tfrecord files found in '{TFRECORD_DIR}'"); return

    print(f"Found {len(tfrecord_files)} files to process.")
    for file_path in tfrecord_files:
        file_basename = os.path.basename(file_path).split('.tfrecord')[0]
        dataset = tf.data.TFRecordDataset(file_path)
        try:
            for i, record in enumerate(dataset):
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                create_animation_for_scenario(scenario, file_basename=file_basename, scenario_index=i + 1)
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_basename}: {e}")
    print("\n--- All scenarios processed. ---")

if __name__ == '__main__':
    main()