import os
from glob import glob
from pathlib import Path
from random import shuffle, seed, randrange

import cv2
from tqdm import tqdm

from mt.file_io.utils import extract_frames

seed(42)

def main(data_folder: str, main_output_folder:str, expert_output_folder:str, num_frames: int = 10):

    video_files = (glob(os.path.join(data_folder, "*.avi")) +
                   glob(os.path.join(data_folder, "*.tif")) +
                     glob(os.path.join(data_folder, "*.mp4")))

    if not video_files:
        print(f"No .avi or .tif videos found in '{data_folder}'")
        return

    print(f"Found {len(video_files)} video files in '{data_folder}'")

    os.makedirs(expert_output_folder, exist_ok=True)
    os.makedirs(main_output_folder, exist_ok=True)

    for video_idx, video_path in enumerate(video_files):
        print(f"Processing: {video_path}")


        frames_list, fps = extract_frames(video_path, num_crops = 10, crop_size=(512, 512))
        if not frames_list:
            print(f"  -> Skipping video, no frames were extracted.")
            return

        base_name = Path(video_path).stem
        # select a single random index from the frames_list
        random_crop_idx = randrange(len(frames_list))

        for crop_idx, cropped_frames in tqdm(enumerate(frames_list), desc=f"Processing {video_idx}/{len(video_files)}", unit="crop"):

            shuffle(cropped_frames)
            cropped_frames_w_idx = list(enumerate(cropped_frames))
            shuffle(cropped_frames_w_idx)
            indices, cropped_frames_shuffled = zip(*cropped_frames_w_idx)

            cropped_frames_shuffled = cropped_frames_shuffled[:num_frames]
            indices = indices[:num_frames]

            crop_output_folder = os.path.join(main_output_folder, f"{base_name}_crop_{crop_idx}")
            os.makedirs(crop_output_folder, exist_ok=True)

            random_frame_idx = randrange(len(cropped_frames_shuffled))

            for frame_idx, frame in enumerate(cropped_frames_shuffled):
                frame_num = indices[frame_idx]

                crop_output_path = os.path.join(crop_output_folder, f"{base_name}_crop_{crop_idx}_frame_{frame_num:02d}.png")
                expert_output_path = os.path.join(expert_output_folder, f"{base_name}_crop_{crop_idx}_frame_{frame_num:02d}.png")

                frame_uint8 = (frame * 255).astype('uint8')
                frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)

                cv2.imwrite(crop_output_path, frame_rgb)

                if (crop_idx == random_crop_idx) and (frame_idx == random_frame_idx):
                    cv2.imwrite(expert_output_path, frame_rgb)



if __name__ == "__main__":

    data_path = os.path.join('data', 'mpi', 'type_B')
    main_output = os.path.join('data', 'SynMT', 'real', 'full')
    expert_output = os.path.join('data', 'SynMT', 'real', 'validation')

    main(data_path, main_output, expert_output)