import cv2
import numpy as np


def diamond_search(curr_block, ref_frame, search_range):
    block_height, block_width = curr_block.shape
    best_match = None
    min_cost = float('inf')

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            candidate_y = curr_block_y + dy
            candidate_x = curr_block_x + dx

            if candidate_y >= 0 and candidate_y + block_height <= ref_frame.shape[0] and \
                    candidate_x >= 0 and candidate_x + block_width <= ref_frame.shape[1]:

                candidate_block = ref_frame[candidate_y:candidate_y + block_height,
                                  candidate_x:candidate_x + block_width]

                cost = np.sum(np.abs(curr_block - candidate_block))

                if cost < min_cost:
                    min_cost = cost
                    best_match = (candidate_x, candidate_y)

    return best_match


# Open the video file
input_video_path = 'input_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)

block_size = 64
search_range = 32

# Create VideoWriter for the output video
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Read the first frame as reference frame
ret, reference_frame = cap.read()
if not ret:
    print("Error reading reference frame")
    exit()

reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, current_frame = cap.read()

    if not ret:
        break

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Motion estimation and compensation
    motion_compensated_frame = np.zeros_like(current_frame_gray)

    for y in range(0, frame_height - block_size + 1, block_size):
        for x in range(0, frame_width - block_size + 1, block_size):
            curr_block = current_frame_gray[y:y + block_size, x:x + block_size]
            curr_block_x = x
            curr_block_y = y

            best_match = diamond_search(curr_block, reference_frame_gray, search_range)
            motion_vector = best_match

            ref_block = reference_frame_gray[motion_vector[1]:motion_vector[1] + block_size,
                        motion_vector[0]:motion_vector[0] + block_size]
            motion_compensated_frame[y:y + block_size, x:x + block_size] = ref_block

    # Display or save motion compensated frame
    out.write(motion_compensated_frame)

# Release the video capture and writer objects
cap.release()
out.release()

cv2.destroyAllWindows()