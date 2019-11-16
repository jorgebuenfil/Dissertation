# !/usr/bin/python3
"""
build_face_dataset.py
Press 's' to save frame, 'q' to quit.
"""
import imutils
from imutils.video import VideoStream
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import time
import cv2
import os

video_source          = 1  # 0 for internal camera; 1 for USB camera
cascade               = \
    "/home/jorge/workspace/OpenCV_projects/613-build-face-dataset/haarcascade_frontalface_default.xml"
directory_destination = \
    r"/home/jorge/Pictures/Facial_Recognition/raw_dataset/"

def get_user_input():
    prompt = Tk()
    Label(prompt, text="Name of person to scan: ").grid(row=0)

    e1 = Entry(prompt)
    e1.grid(row=0, column=1)
    Button(prompt, text='Accept', command=prompt.quit).grid(row=3, column=0, sticky=W, pady=4)
    mainloop()
    return e1.get()

def show_message(title, message, width=600):
    message_window = tk.Tk()
    message_window.title(title)
    msg = tk.Message(message_window, text=message, width=width)
    msg.config(bg='lightgreen', font=('times', 12, 'italic'))
    msg.pack()

def main():
    total = 0

    # load OpenCV's Haar cascade for face detection from disk
    detector = cv2.CascadeClassifier(cascade)

    show_message("[INFO]", "Enter the name of the user in the next window.\n"
                           "The video stream will start after that.\n"
                           "Press 's' to save an image or 'q' to quit")

    user_name = get_user_input()
    output_destination = directory_destination + user_name

    # Initialize video stream; warm up camera sensor and initialize the total number of example faces written to disk
    vs = VideoStream(src=1).start()

    # If using the Raspberry Pi use the following instruction instead:
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        """
        Grab the frame from the threaded video stream, clone it, (just in case we want to write it to disk), 
        and then resize the frame to apply face detection faster.
        """
        frame    = vs.read()
        original = frame.copy()
        frame    = imutils.resize(frame, width=400)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # loop over the face detections and draw them on the frame
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `s` key was pressed, write the *original* frame to disk so we can later
        # process it and use it for face recognition
        if key == ord("s"):
            filename = r"{}.png".format(str(total).zfill(3))
            path = output_destination + filename
            cv2.imwrite(path, original)
            total += 1

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break131113


    # cleanup
    message = str(total) + " face images stored in " + output_destination
    show_message("Summary", message)
    #print("[INFO] {} face images stored".format(total))
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()
