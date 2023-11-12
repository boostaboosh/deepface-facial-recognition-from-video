# Import DeepFace class for facial recognition from the deepface library.
from deepface import DeepFace
# Import OpenCV (cv2) library for video processing.
import cv2


# This function takes a reference image and an input image as input and checks if the input image contains a face which
# matches a face in the reference image.
def is_match(face_image, input_image):
    # We use the DeepFace library to verify if the face in the image matches with the reference image:
    # "reference.jpeg". The distance metric is the measure of similarity between two feature vectors. A feature
    # vector is a list of numbers that represent the features of a face. The 'cosine' distance metric measures the
    # cosine of the angle between two feature vectors. If the vectors are identical, the angle is zero degrees and
    # the cosine is one, meaning the faces are the same. If the vectors are completely different the cosine is 90
    # degrees and the cosine is zero, meaning the faces are different. The enforce_detection = False parameter means an
    # error will not be raised if deepface fails to detect a face in either the reference or input images. Instead, with
    # enforce_detection set to False if no face is detected in the reference or input image it will still attempt a
    # comparison even if face detection fails for some reason.
    result = DeepFace.verify(face_image, input_image, model_name="Facenet512", distance_metric="cosine",
                             enforce_detection=False)

    # The result is a dictionary with a key 'verified' that is True if it's a match and False otherwise.
    return result["verified"]


reference_face_image = "superman.jpeg"
# We start the webcam with cv2.VideoCapture. 0 means the default webcam. 1 means the usb webcam. The VideoCapture
# method is used to capture video from a camera or a file. The method takes an argument that specifies the camera
# index or the name of the video file.
capture = cv2.VideoCapture(1)
# Now, we continuously get frames from the video, detect face, and draw squares around them.
while True:
    # We read a new frame from the video. The method capture.read() returns two values:
    # (1) the boolean value that indicates whether the frame was successfully read or not
    # and
    # (2) the image, known as frame, captured from the video.
    read_successfully, captured_video_frame = capture.read()

    # If the frame was properly read.
    if read_successfully:
        # We detect faces in the frame. The extract_faces function detects faces in an image. The detector_backend
        # parameter specifies the algorithm that the function uses to detect faces. There are several algorithms
        # available, including OpenCV, SSD, Dlib, MTCNN, FasterMTCNN, RetinaFace, MediaPipe, YOLOv8 Face,
        # and YuNet which are all wrapped in the deepface library. OpenCV is the default detector. RetinaFace and
        # MTCNN over perform in face detection and alignment phases, but they are much slower. If the speed of
        # detection is important then you should use opencv or ssd. If accuracy of detection is important then you
        # should use retinaface or mtcnn.
        detected_faces = DeepFace.extract_faces(captured_video_frame, detector_backend="retinaface")

        # For each face detected in the frame.
        for detected_face in detected_faces:
            # We get the coordinates of the square around the face.
            x, y, w, h = (detected_face["facial_area"]["x"], detected_face["facial_area"]["y"],
                          detected_face["facial_area"]["w"], detected_face["facial_area"]["h"])

            # We check if the face matches with the reference face image
            if is_match(reference_face_image, detected_face["face"]):
                # If the detected face matches the reference face image, we draw a green square around the face.
                # The color tuple in Open CVs' cv2.rectangle uses the format (B, G, R), which corresponds to the order
                # of the Blue, Green, and Red color channels.
                color = (0, 255, 0)
            else:
                # If the detected face does not match the reference face image, we draw a red square around the face.
                # The color tuple in Open CVs' cv2.rectangle uses the format (B, G, R), which corresponds to the order
                # of the Blue, Green, and Red color channels.
                color = (0, 0, 255)

            # Draw a square on the frame.
            cv2.rectangle(captured_video_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

        # Display the frame with the squares. The imshow method from the open cv 2 library displays an image in a
        # window. Like showing a picture on a screen.
        cv2.imshow("Webcam Video Frame Capture", captured_video_frame)

    # If 'q' (q stand for quit) is pressed on the keyboard, stop the loop.
    # The waitKey method from the open cv 2 library waits for the time specified in milliseconds for a key event to
    # occur. If a key is pressed the waitKey(milliseconds) method returns the ASCII value of the key.
    # If no key is pressed, it returns -1.
    pressedKey = cv2.waitKey(1)
    print(pressedKey)
    # The ord() function gets the Unicode encoding of a character.
    # The Unicode code point for the letter ‘q’ is 113. When you call ord('q'), it returns 113.
    # For backward compatibility purposes, the first 128 Unicode characters point to ASCII characters.  So, we can say
    # that ASCII is a subset of Unicode.
    if pressedKey == ord('q'):
        break

# Release the webcam.
capture.release()

# Destroy all windows.
cv2.destroyAllWindows()