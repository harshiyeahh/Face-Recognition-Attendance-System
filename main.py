import cv2
import numpy as np
import sqlite3
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

labels = np.load(
    'trainer/labels.npy',
    allow_pickle=True
).item()

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

def mark_attendance(name):

    conn = sqlite3.connect('attendance.db')

    cursor = conn.cursor()

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
        (name, date, time)
    )

    conn.commit()
    conn.close()

marked_names = []

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        1.2,
        5
    )

    for (x, y, w, h) in faces:

        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        if confidence < 70:

            name = labels[id]

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            if name not in marked_names:

                mark_attendance(name)

                marked_names.append(name)

        else:

            cv2.putText(
                frame,
                "Unknown",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()