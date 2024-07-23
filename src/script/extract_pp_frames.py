import os
import subprocess
import sys

def extract_frames_from_videos(folder_path, s, e):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Create an output directory for frames if it does not exist
    output_folder = os.path.join(folder_path, 'frames')
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter out video files (assuming .mp4 extension here; modify if needed)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files.sort()
    video_files = video_files[s:s+e]
    print(len(video_files))

    if not video_files:
        print("No video files found in the directory.")
        return

    # Process each video file
    for i, video_file in enumerate(video_files):
        print('===============>', i, len(video_files))

        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Create a directory for the current video's frames
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Command to extract frames using ffmpeg
        command = [
            'ffmpeg', '-i', video_path, 
            os.path.join(video_output_folder, f"%07d.jpg")
        ]

        # Run the command
        subprocess.run(command)

    print("Frame extraction completed.")

folder_path = '' # path to personpath raw_videos, e.g., dataset/personpath22/raw_data'
start_idx, end_idx = sys.argv[1], sys.argv[2]
start_idx, end_idx = int(start_idx), int(end_idx)
print(start_idx, end_idx)
extract_frames_from_videos(folder_path, start_idx, end_idx)

# Example command
# python extract_pp_frames.py 0 250
