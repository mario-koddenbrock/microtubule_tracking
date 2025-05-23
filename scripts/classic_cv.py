import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from glob import glob

# --- Parameters ---
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 5
LINE_ASSOCIATION_THRESHOLD = 20  # pixels

# --- Helper functions ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(denoised, 50, 150)
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
def process_video(image_folder):
    image_files = sorted(glob(os.path.join(image_folder, "*.png")))
    tracked_lengths = []
    video_frames = []

    prev_centers = None
    microtubule_tracks = {}

    for frame_idx, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            continue

        vis_frame = frame.copy()
        edges = preprocess_frame(frame)
        lines = detect_lines(edges)

        curr_centers = np.array([compute_line_center(line) for line in lines])
        curr_lengths = [compute_line_length(line) for line in lines]

        if prev_centers is not None and len(curr_centers) > 0:
            associations = associate_lines(prev_centers, curr_centers)
            for old_idx, new_idx in associations:
                microtubule_id = old_idx
                if microtubule_id not in microtubule_tracks:
                    microtubule_tracks[microtubule_id] = []
                microtubule_tracks[microtubule_id].append((frame_idx, curr_lengths[new_idx]))
        else:
            for i, length in enumerate(curr_lengths):
                microtubule_tracks[i] = [(frame_idx, length)]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        video_frames.append(vis_frame)
        prev_centers = curr_centers

    return video_frames, microtubule_tracks

# --- Visualization ---
def save_video(frames, output_path, fps=5):
    if not frames:
        print("No frames to save.")
        return
    height, width, _ = frames[0].shape
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
    image_folder = "path/to/your/images"  # Change this to your image folder
    video_output = "microtubule_tracking.mp4"
    plot_output = "microtubule_lengths.png"

    frames, tracks = process_video(image_folder)
    save_video(frames, video_output)
    plot_lengths(tracks, plot_output)

    print("Tracking complete.")
    print(f"Video saved to: {video_output}")
    print(f"Plot saved to: {plot_output}")
