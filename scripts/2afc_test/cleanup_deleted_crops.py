import os
import shutil
import re

def get_crop_name_from_filename(filename):
    """
    Extracts the crop name (e.g., '1_8pt5uMporcTub001_crop_4') from a filename.
    Example: '1_8pt5uMporcTub001_crop_4_frame_233.png' -> '1_8pt5uMporcTub001_crop_4'
    """
    match = re.match(r'(.+)_frame_\d+\.png$', filename)
    if match:
        return match.group(1)
    return None

def main():
    """
    Identifies manually deleted crops from the single_frame folder and deletes
    corresponding folders from the all_frames data folder.
    """
    all_frames_path = os.path.join('data', 'SynMT', 'real', 'full', 'all_frames')
    single_frame_output = os.path.join('data', 'SynMT', 'real', 'full', 'single_frame')

    if not os.path.isdir(all_frames_path):
        print(f"Error: 'all_frames' directory not found at: {all_frames_path}")
        return

    if not os.path.isdir(single_frame_output):
        print(f"Error: 'single_frame' directory not found at: {single_frame_output}")
        return

    # Get all crop folders from the 'all_frames' directory
    all_crop_folders = {item for item in os.listdir(all_frames_path) if os.path.isdir(os.path.join(all_frames_path, item))}

    # Get all unique crop names present in the 'single_frame' directory
    single_frame_files = os.listdir(single_frame_output)
    present_single_frame_crops = set()
    for filename in single_frame_files:
        crop_name = get_crop_name_from_filename(filename)
        if crop_name:
            present_single_frame_crops.add(crop_name)

    # Determine which crop folders are in 'all_frames' but have no corresponding files in 'single_frame'
    deleted_crops = all_crop_folders - present_single_frame_crops

    if not deleted_crops:
        print("No crops appear to have been deleted from the single_frame folder.")
        print("The 'all_frames' directory is already consistent with the 'single_frame' directory.")
        return

    print("The following crops appear to be deleted from the single_frame folder:")
    for crop in sorted(list(deleted_crops)):
        print(f"- {crop}")

    # Delete the corresponding folders from the 'all_frames' path
    print("\nDeleting corresponding folders from the 'all_frames' directory...")
    for crop_folder in deleted_crops:
        folder_to_delete = os.path.join(all_frames_path, crop_folder)
        try:
            shutil.rmtree(folder_to_delete)
            print(f"Deleted: {folder_to_delete}")
        except OSError as e:
            print(f"Error deleting {folder_to_delete}: {e}")

    print("\nCleanup complete.")

if __name__ == '__main__':
    main()

