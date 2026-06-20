import cv2
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

user_id = input("Enter User ID: ")
name = input("Enter Name: ")

path = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        file_name = f"{path}/{count}.jpg"

        cv2.imwrite(file_name, face)

        count += 1

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) == 13 or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print("Face Capture Completed")