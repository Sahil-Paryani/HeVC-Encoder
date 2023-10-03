import cv2
import numpy as np

def process_frame(reference_frame, current_frame, block_size, search_range, threshold):
    height, width, _ = reference_frame.shape
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size

    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2))

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            min_mse = float('inf')
            best_motion_vector = [0, 0]

            current_block = current_frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_y = int(min(max(y * block_size + dy, 0), height - block_size))
                    ref_x = int(min(max(x * block_size + dx, 0), width - block_size))

                    reference_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                    mse = np.sum(np.square(reference_block - current_block))

                    if mse < min_mse:
                        min_mse = mse
                        best_motion_vector = [dy, dx]

            if min_mse > threshold:
                motion_vectors[y, x] = best_motion_vector

    motion_compensated_frame = np.zeros_like(current_frame)
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            motion_vector = motion_vectors[y, x]
            ref_y = int(min(max(y * block_size + motion_vector[0], 0), height - block_size))
            ref_x = int(min(max(x * block_size + motion_vector[1], 0), width - block_size))

            motion_compensated_frame[y * block_size:(y + 1) * block_size,
            x * block_size:(x + 1) * block_size] = reference_frame[ref_y:ref_y + block_size,
                                                   ref_x:ref_x + block_size]

    residual_image = current_frame - motion_compensated_frame
    result_image = motion_compensated_frame + residual_image
    error = current_frame - result_image

    return motion_compensated_frame, residual_image, result_image, error, motion_vectors


def main(input_video_path, output_video_path, block_size, search_range, threshold):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    fourcc = cv2.VideoWriter_fourcc(*'VID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    reference_frame = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if reference_frame is None:
            reference_frame = frame
            frame_number += 1
            continue

        print(f"Processing frame {frame_number}...")
        motion_compensated_frame, _, result_image, _ = process_frame(reference_frame, frame, block_size, search_range, threshold)
        out.write(result_image.astype(np.uint8))

        reference_frame = frame
        frame_number += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = "vector_football.mp4"
    output_video_path = "vector_football.mov"
    block_size = 64
    search_range = 64
    threshold = 0

    main(input_video_path, output_video_path, block_size, search_range, threshold)