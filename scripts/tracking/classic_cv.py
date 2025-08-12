import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from mt.file_io.utils import extract_frames

# --- Parameters ---
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 5
LINE_ASSOCIATION_THRESHOLD = 20  # pixels
MIN_EDGE_PERIMETER = 10 # pixels

# --- Helper functions ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    denoised = cv2.GaussianBlur(gray, (5, 5), 1)

    v = np.median(denoised)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(denoised, lower, upper)

    # remove small contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.arcLength(cnt, True) < MIN_EDGE_PERIMETER:
            cv2.drawContours(edges, [cnt], 0, 0, -1)  # Remove contour from edge map

    return edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    return lines if lines is not None else []

def compute_line_center(line):
    x1, y1, x2, y2 = line[0]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def compute_line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def associate_lines(prev_centers, curr_centers):
    cost_matrix = np.linalg.norm(prev_centers[:, None] - curr_centers[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    associations = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < LINE_ASSOCIATION_THRESHOLD:
            associations.append((i, j))
    return associations


# --- Main processing ---
def process_frames(frames):
    video_frames = []
    microtubule_tracks = {}
    prev_centers = None
    next_id = 0
    prev_id_map = {}

    for frame_idx, frame in tqdm(enumerate(frames), "Processing frames", total=len(frames)):
        vis_frame = frame.copy()
        edges = preprocess_frame(frame)
        lines = detect_lines(edges)

        curr_centers = np.array([compute_line_center(line) for line in lines]) if len(lines) > 0 else np.empty((0, 2))
        curr_lengths = [compute_line_length(line) for line in lines] if len(lines) > 0 else []

        curr_id_map = {}

        if prev_centers is not None and len(curr_centers) > 0:
            associations = associate_lines(prev_centers, curr_centers)
            for old_idx, new_idx in associations:
                track_id = prev_id_map.get(old_idx, next_id)
                curr_id_map[new_idx] = track_id
                if track_id not in microtubule_tracks:
                    microtubule_tracks[track_id] = []
                    next_id += 1
                microtubule_tracks[track_id].append((frame_idx, curr_lengths[new_idx]))

        for i, length in enumerate(curr_lengths):
            if i not in curr_id_map:
                curr_id_map[i] = next_id
                microtubule_tracks[next_id] = [(frame_idx, length)]
                next_id += 1

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            track_id = curr_id_map.get(i, -1)
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        video_frames.append(vis_frame)
        prev_centers = curr_centers
        prev_id_map = curr_id_map

    return video_frames, microtubule_tracks

# --- Visualization ---
def save_video(frames, output_path, fps=5):
    if not frames:
        print("No frames to save.")
        return
    if len(frames[0].shape) == 3:
        height, width, _ = frames[0].shape
    else:
        height, width = frames[0].shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

def plot_lengths(microtubule_tracks, output_path):
    plt.figure(figsize=(10, 6))
    for track_id, values in microtubule_tracks.items():
        frames, lengths = zip(*values)
        plt.plot(frames, lengths, label=f'Track {track_id}')
    plt.xlabel("Frame")
    plt.ylabel("Length (pixels)")
    plt.title("Microtubule Length Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Run everything ---
if __name__ == "__main__":

    data_path = os.path.join('data', 'mpi')
    output_path = os.path.join('results', 'mpi')

    os.makedirs(output_path, exist_ok=True)

    video_files = glob(os.path.join(data_path, "*.avi")) + glob(os.path.join(data_path, "*.tif"))

    for video_path in video_files:
        print(f"Processing: {video_path}")
        frames, _ = extract_frames(video_path)
        tracked_frames, tracks = process_frames(frames)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output = os.path.join(output_path, f"{base_name}_tracking.mp4")
        plot_output = os.path.join(output_path, f"{base_name}_lengths.png")

        save_video(tracked_frames, video_output)
        # plot_lengths(tracks, plot_output)

        print(f"Video saved to: {video_output}")
        # print(f"Plot saved to: {plot_output}")
