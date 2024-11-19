import cv2
import os
import argparse
parser= argparse.ArgumentParser()
parser.add_argument('video',type=str)
args = parser.parse_args()
# Input video path
input_video_path = args.video
output_dir = f"output_segments_{input_video_path.split('.')[0]}"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

# Load the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Calculate the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Process video in chunks of 60 frames
chunk_size = 30
chunk_index = 0

while True:
    frames = []  # To store the current chunk of frames
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        break  # No more frames to process

    # Save the chunk as a new video
    output_path = os.path.join(output_dir, f"segment_{chunk_index}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()
    chunk_index += 1

cap.release()
print(f"Video split into {chunk_index} segments of {chunk_size} frames each.")

video_name = input_video_path.split('.')[0]
file = open(f'{video_name}_correct.txt')
actions = file.readlines()
file.close()
new = []
value = 0
os.makedirs(f'output_segments_text_{video_name}',exist_ok=True)
for index, action in enumerate(actions):
    new.append(action.strip())    

    if (((index+1)%30) ==0):
        naming = f"segment_{value}"+".txt"
        with open(f"output_segments_text_{video_name}/{naming}", "w") as file:
            file.write("\n".join(new))
        file.close()
        new = []
        value+=1
        print(f"saving action# {index}")
