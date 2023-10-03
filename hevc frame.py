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
    print(motion_vectors)
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
    # Convert the color image to a format suitable for DCT (e.g., YCrCb)
    # yuv_image = cv2.cvtColor(residual_image, cv2.COLOR_BGR2YCrCb)

    # y_channel = yuv_image[:, :, 0]
    # cr_channel = yuv_image[:, :, 1]
    # cb_channel = yuv_image[:, :, 2]

    # dct_y_channel = cv2.dct(np.float32(y_channel))
    # dct_cr_channel = cv2.dct(np.float32(cr_channel))
    # dct_cb_channel = cv2.dct(np.float32(cb_channel))

    # Display the DCT-transformed channels
    # cv2.imshow('DCT Y Channel', dct_y_channel)
    # cv2.imshow('DCT Cr Channel', dct_cr_channel)
    # cv2.imshow('DCT Cb Channel', dct_cb_channel)

    return motion_compensated_frame, residual_image, result_image, error

def main():
    reference_frame = cv2.imread("frame_0263.png")
    current_frame = cv2.imread("frame_0262.png")
    frame1 = reference_frame
    frame2 = current_frame
    block_size = 64
    search_range = 64
    threshold = 0

    while block_size >= 4:  # Continue until reaching 4x4 blocks
        motion_compensated_frame, residual_image, result_image, error = process_frame(reference_frame, current_frame, block_size, search_range, threshold)

        # Create named windows with a fixed size
        cv2.namedWindow("Reference Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Motion Compensated Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Residual Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Error", cv2.WINDOW_NORMAL)

        cv2.imshow("Reference Frame", reference_frame)
        cv2.imshow("Current Frame", current_frame)
        cv2.imshow("Motion Compensated Frame", motion_compensated_frame)
        cv2.imshow("Residual Image", residual_image)
        cv2.imshow("Final Image", result_image)
        cv2.imshow("Error", error)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()