import cv2
import numpy as np

def diamond_search(reference_frame, current_block, search_range):
    best_mad = float('inf')
    best_mv = (0, 0)

    diamond_search_steps = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    for step_x, step_y in diamond_search_steps:
        ref_x = x + step_x
        ref_y = y + step_y
        if ref_x < 0 or ref_x + block_size >= width or ref_y < 0 or ref_y + block_size >= height:
            continue

        reference_block = reference_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
        mad = np.sum(np.abs(current_block - reference_block))

        if mad < best_mad:
            best_mad = mad
            best_mv = (step_x, step_y)

    return best_mv

# Load video frames
video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


block_size = 16
search_range = 16

class YourX265Encoder:
    def encode_block(self, block, mv):
        # Your encoding logic here
        pass

encoder = YourX265Encoder()

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'X264')  # Choose the codec (X264, XVID, etc.)
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

ret, reference_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    encoded_frame = np.zeros_like(frame)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            current_block = frame[y:y+block_size, x:x+block_size]

            # Perform diamond search
            best_mv = diamond_search(reference_frame, current_block, search_range)

            # Encode the current block using the best motion vector
            encoded_block = encoder.encode_block(current_block, best_mv)

            # Store the encoded block in the output frame
            encoded_frame[y:y+block_size, x:x+block_size] = encoded_block

    reference_frame = frame
    out.write(encoded_frame)

    input_size = (frame.shape[1], frame.shape[0])
    output_size = (encoded_frame.shape[1], encoded_frame.shape[0])
    print(f"Input Video Size: {input_size[0]}x{input_size[1]}, Output Video Size: {output_size[0]}x{output_size[1]}")

# close windows
cap.release()
out.release()
cv2.destroyAllWindows()
