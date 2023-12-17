import cv2
from deepface import DeepFace

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            resized_face_frame = cv2.resize(face_frame, (160, 160))

            try:
                results = DeepFace.analyze(resized_face_frame, actions=['emotion'], enforce_detection=False)

                # Check the type of results
                if isinstance(results, list):
                    # Extract the first item if results is a list
                    result = results[0]
                elif isinstance(results, dict):
                    # Use results directly if it's a dictionary
                    result = results
                else:
                    raise TypeError("Unexpected result format from DeepFace.analyze")

                print("Result:", result)  # Print the result for debugging

                if 'dominant_emotion' in result:
                    emotion = result['dominant_emotion']
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, emotion, (x, y-10), font, 1, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            except Exception as e:
                print("Error in emotion detection:", e)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
