## detection-and-tracking-people
This project uses TensorFlow Hub's EfficientDet model to detect objects in a video, specifically targeting crowd scenarios. The processed video is saved with bounding boxes around detected individuals, and individual frames can also be saved if desired.

# Features
Loads a pre-trained EfficientDet model from TensorFlow Hub.
Processes input video to detect objects.
Draws bounding boxes around detected people.
Saves the annotated video and optionally saves individual frames.

# Requirements
TensorFlow
TensorFlow Hub
OpenCV
NumPy
