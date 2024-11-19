import cv2, os
import argparse
parser= argparse.ArgumentParser()
parser.add_argument('video',type=str)
args = parser.parse_args()
video = args.video
video_name = video.split('.')[0]

text_file_path = f"{video_name}_correct.txt"

try:
        # Read text lines from the file
    with open(text_file_path, "r") as file:
        text_lines = [line.strip() for line in file.readlines()]
except:
    raise Exception(f"No text file for video{video_name} present in the current dir")


# Paths to input video and text file
input_video_path = video
output_video_path = f"{video_name}_vis.mp4"

# Read text lines from the file
with open(text_file_path, "r") as file:
    text_lines = [line.strip() for line in file.readlines()]

# Open the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame and add text
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Add text from the file to the frame
    text = text_lines[frame_idx % len(text_lines)]  # Loop through text if frames exceed lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 50)  # Text position
    font_scale = 1
    font_color = (0, 255, 0)  # Green
    thickness = 2

    # Overlay the text on the frame
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    frame_idx += 1

# Release resources
cap.release()
out.release()

print(f"Video saved to {output_video_path}")