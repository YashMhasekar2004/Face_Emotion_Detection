import cv2
from keras.models import model_from_json
import numpy as np

# Load the trained model
json_file = open("facialemotionmodel.json", "r")  # Make sure the JSON file is correct
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the correct model weights
model.load_weights("facialemotionmodel.h5")  # Updated to your trained model

# Haar cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess image for model prediction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to model's expected input size
    return feature / 255.0  # Normalize pixel values to [0, 1]

# Function to detect emotion from an image
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    print(f"Image dimensions: {gray.shape}")  # Debugging: check image size
    
    # Adjusted Haar Cascade parameters for better face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces detected.")  # Debug: Check if faces are being detected
        return image
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Crop the face from the image
        face = cv2.resize(face, (48, 48))  # Resize to 48x48 for the model
        face = extract_features(face)  # Extract features for the model
        
        # Get model prediction
        pred = model.predict(face)
        print("Prediction:", pred)  # Debug: Print the raw prediction output

        # Get the emotion label
        emotion = labels[pred.argmax()]
        print("Detected Emotion:", emotion)  # Debug: Check the detected emotion

        # Draw a rectangle around the face and label the emotion
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image

# Function to detect emotion from a camera feed
def detect_emotion_from_camera():
    webcam = cv2.VideoCapture(1)  # Start video capture
    if not webcam.isOpened():
        print("Error: No camera found.")
        return
    
    print("Camera is opened. Press 'q' to exit.")
    while True:
        ret, frame = webcam.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to capture image from camera.")
            break
        
        output_frame = detect_emotion(frame)  # Process the frame for emotion detection
        cv2.imshow("Emotion Detection - Camera", output_frame)  # Show the processed frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break
    
    webcam.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Option 2: Use an existing image for emotion detection
def detect_emotion_from_image(image_path):
    image = cv2.imread(image_path)  # Load image
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    print(f"Loaded image from {image_path}")  # Debug: Check if image is loaded
    output_image = detect_emotion(image)
    cv2.imshow("Emotion Detection - Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    option = input("Choose mode (1: Camera, 2: Image): ")
    if option == '2':
        image_path = input("Enter image file path: ")
        print(f"Using image path: {image_path}")  # Debug: Print the image path
        detect_emotion_from_image(image_path)
    elif option == '1':
        detect_emotion_from_camera()  # Enable camera detection
    else:
        print("Invalid option. Please select 1 or 2.")
