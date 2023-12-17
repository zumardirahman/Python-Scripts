import cv2
from deepface import DeepFace

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_frame = frame[y:y+h, x:x+w]

            try:
                results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)

                # Handling different types of results
                if isinstance(results, list):
                    result = results[0]  # Assuming the first result is relevant
                elif isinstance(results, dict):
                    result = results
                else:
                    raise ValueError("Unexpected result type from DeepFace.analyze")

                print(result)  # Log the result for debugging

                emotion = result.get('dominant_emotion', 'N/A')
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (x, max(y - 10, 0))
                cv2.putText(frame, emotion, text_position, font, 1, (0, 255, 0), 2)

            except Exception as e:
                print("Error in emotion detection:", e)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
