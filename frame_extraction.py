import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=1, image_format='png'):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input video file (.mp4, .avi, etc.).
        output_folder (str): Path to the directory where frames will be saved.
        frame_interval (int): Extract every Nth frame (default is 1 for all frames).
                             Set > 1 to skip frames (e.g., 30 for approx 1 frame/sec
                             on a 30fps video).
        image_format (str): Format to save frames ('png' or 'jpg'). 'png' is lossless,
                           'jpg' is lossy but smaller file size.
    """
    # --- 1. Validation and Setup ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        print(f"Creating output directory: '{output_folder}'")
        os.makedirs(output_folder)
    else:
        print(f"Output directory '{output_folder}' already exists. Files may be overwritten.")

    if image_format not in ['png', 'jpg']:
        print(f"Warning: Invalid image format '{image_format}'. Defaulting to 'png'.")
        image_format = 'png'

    # --- 2. Open Video File ---
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # --- 3. Get Video Properties (Informative) ---
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    print("-" * 30)
    print(f"Video Info:")
    print(f"  Path: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration_sec:.2f} seconds")
    print(f"  Output Folder: {output_folder}")
    print(f"  Frame Interval: {frame_interval} (saving every {frame_interval} frame(s))")
    print(f"  Image Format: .{image_format}")
    print("-" * 30)


    # --- 4. Frame Extraction Loop ---
    frame_count_read = 0
    frame_count_saved = 0
    success = True

    while success:
        # Read the next frame
        success, frame = video_capture.read()

        if not success:
            # End of video or error reading frame
            break

        # Check if this frame should be saved based on the interval
        if frame_count_read % frame_interval == 0:
            # Construct the output filename
            # Using padding (e.g., 06d) ensures files sort correctly alphabetically
            # Adjust padding based on expected number of frames (e.g., 07d for >999,999 frames)
            frame_filename = os.path.join(output_folder, f"frame_{frame_count_saved:06d}.{image_format}")

            # Save the current frame as an image file
            cv2.imwrite(frame_filename, frame)
            frame_count_saved += 1

            # Optional: Print progress indicator
            if frame_count_saved % 100 == 0:
                print(f"Saved frame {frame_count_saved}...")

        frame_count_read += 1

    # --- 5. Release Resources ---
    video_capture.release()
    # cv2.destroyAllWindows() # Usually not needed unless you displayed frames

    print("-" * 30)
    print(f"Finished processing.")
    print(f"Total frames read from video: {frame_count_read}")
    print(f"Total frames saved to '{output_folder}': {frame_count_saved}")
    print("-" * 30)