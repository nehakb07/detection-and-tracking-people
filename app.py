import os
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: All messages, 1: Filter out INFO messages, 2: Filter out WARNING messages, 3: Filter out ERROR messages

# Load the model
MODEL_PATH = 'https://tfhub.dev/tensorflow/efficientdet/d0/1'

def load_model(model_path):
    """Load the object detection model from TensorFlow Hub."""
    model = hub.load(model_path)
    print("Available signatures:", list(model.signatures.keys()))
    return model

model = load_model(MODEL_PATH)

def detect_objects(frame, model):
    """Pass a frame through the object detection model."""
    # Convert the frame to uint8 and resize to match model's expected input dimensions
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    frame_rgb = frame_rgb[tf.newaxis, ...]  # Add batch dimension
    
    # Run inference
    results = model(frame_rgb)
    return results

def draw_bboxes(frame, data, threshold=0.5):
    """Draw bounding boxes on the frame."""
    im_height, im_width, _ = frame.shape
    boxes = data['detection_boxes'].numpy()[0]
    classes = data['detection_classes'].numpy()[0]
    scores = data['detection_scores'].numpy()[0]
    num_detections = int(data['num_detections'].numpy()[0])

    for i in range(num_detections):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
    return frame

def process_video(input_path, output_path, model, frames_per_second=4, detection_threshold=0.5, save_frames=False):
    """Process the video file, extract frames, detect people, save annotated video and frames."""
    
    # Create output folder if it doesn't exist
    output_folder = 'output'
    if save_frames and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    saved_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = detect_objects(frame, model)
            frame_with_bboxes = draw_bboxes(frame, results, detection_threshold)
            out.write(frame_with_bboxes)

            # Save frame to the output folder
            if save_frames:
                frame_filename = f'{output_folder}/frame_{saved_frame_count:05d}.jpg'
                cv2.imwrite(frame_filename, frame_with_bboxes)
                saved_frame_count += 1

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output paths
input_video_path = 'crowd/crowd3.mp4'
output_video_path = 'crowd/output3.mp4'

# Process the video and save frames
process_video(input_video_path, output_video_path, model, frames_per_second=4, detection_threshold=0.5, save_frames=True)

print(f"Processed video saved to {output_video_path} and frames saved to 'crowd/output3/' folder")
