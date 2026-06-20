# Face Recognition Attendance System

## Overview

A Python-based attendance management system that uses facial recognition to automatically identify individuals and record attendance.

## Features

- Face image collection using webcam
- Face detection using OpenCV
- Face recognition using LBPH algorithm
- Automatic attendance logging
- SQLite database storage
- Attendance report generation in CSV format

## Technologies Used

- Python
- OpenCV
- NumPy
- SQLite
- Pillow

## Project Workflow

1. Capture face images using webcam.
2. Train the face recognition model.
3. Detect and recognize faces in real time.
4. Automatically mark attendance.
5. Export attendance reports.

## Files

- `capture_faces.py` – Collects face images.
- `train_model.py` – Trains the recognition model.
- `main.py` – Performs face recognition and attendance marking.
- `create_database.py` – Creates SQLite attendance database.
- `attendance_report.py` – Generates attendance reports.

## Future Improvements

- GUI Dashboard
- Cloud Database Integration
- Email Notifications
- Improved Recognition Accuracy
