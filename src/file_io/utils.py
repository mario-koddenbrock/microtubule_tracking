import cv2
import tifffile

def extract_frames(video_path):
    frames = []
    if video_path.lower().endswith(".avi"):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif video_path.lower().endswith(".tif"):
        data = tifffile.imread(video_path)
        for i in range(data.shape[0]):
            frame = (data[i, 1, :, :] / 65535.0 * 255).astype(np.uint8)
            frames.append(frame)
    return frames