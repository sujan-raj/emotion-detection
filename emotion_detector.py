import sys
import subprocess

def install_packages():
    print("========================================")
    print("Checking and installing required packages")
    print("This might take a few minutes on first run...")
    print("========================================")
    packages = ["opencv-python", "deepface", "tf-keras"]
    for pkg in packages:
        try:
            if pkg == 'opencv-python':
                import cv2
            elif pkg == 'tf-keras':
                import tf_keras
            else:
                __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
            print(f"Successfully installed {pkg}")

install_packages()

import cv2
from deepface import DeepFace
import time

def main():
    print("========================================")
    print("Starting Emotion Detection Mini-Project")
    print("========================================")
    print("Loading AI Models... (this may take a few seconds on first run)")
    
    # Initialize webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not access the webcam.")
        print("Please ensure your webcam is connected and accessible.")
        return

    # Load OpenCV's pre-trained Haar Cascade for face detection
    # Concept: Classic Digital Image Processing (DIP) Object Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Webcam started successfully! Press 'q' on your keyboard to exit.")
    
    frame_count = 0
    current_emotion = "Analyzing..."

    while True:
        # 1. Video Frame Extraction
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break
            
        frame_count += 1
        
        # Optionally mirror the frame for a more natural mirror-like viewing experience
        frame = cv2.flip(frame, 1)

        # 2. Color Space Conversion (DIP Step)
        # Convert frame to grayscale for faster face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Object Detection (DIP Step)
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            # Draw a blue bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 4. Image Cropping (DIP Step)
            # Extract the Region of Interest (ROI) containing the face
            face_roi = frame[y:y+h, x:x+w]

            # 5. AI Classification
            # We run the AI model every 3 frames to keep the video feed smooth
            if frame_count % 3 == 0:
                try:
                    # Pass the cropped face to DeepFace's Convolutional Neural Network
                    # enforce_detection=False prevents errors if DeepFace's internal detector fails
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    # DeepFace returns a list of dictionaries if multiple faces are found, 
                    # but we are passing cropped single faces, so we just take the first element.
                    if isinstance(result, list):
                        result = result[0]
                        
                    current_emotion = result['dominant_emotion']
                except Exception as e:
                    # If it fails to analyze, keep the last known emotion
                    pass
            
            # Put the emotion text above the bounding box
            text = f"{current_emotion.capitalize()}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame in a window
        cv2.imshow('Mini Project: Emotion Detection (AI)', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting application...")
            break

    # Clean up: Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Project successfully closed.")

if __name__ == '__main__':
    main()
