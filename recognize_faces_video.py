# !/usr/bin/python3
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

encodings_file   = "/home/jorge/workspace/OpenCV_projects/face_recognition/encodings.pickle"
detection_method = "hog" # choice of 'hog' or 'cnn'
output           = None

def main():
    # load the known faces and embeddings
    data = pickle.loads(open(encodings_file, "rb").read())

    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    vs = VideoStream(src=1).start()
    writer = None
    time.sleep(2.0)

    # loop over frames from the video file stream
    while True:
        frame = vs.read()

        # convert the input frame from BGR to RGB then resize it to have a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r   = frame.shape[1] / float(rgb.shape[1])

        '''
        Detect the (x, y)-coordinates of the bounding boxes corresponding to each face 
        in the input frame, then compute the facial embeddings for each face.
        '''
        boxes     = face_recognition.face_locations(rgb, model = detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names     = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"],	encoding)
            name    = "Unknown"

            # check to see if we have found a match
            if True in matches:
                '''
                Find the indexes of all matched faces then initialize a dictionary to count 
                the total number of times each face was matched
                '''
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                '''
                Determine the recognized face with the largest number of votes 
                (note: in the event of an unlikely tie Python will select first entry in the dictionary)
                '''
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top    = int(top    * r)
            right  = int(right  * r)
            bottom = int(bottom * r)
            left   = int(left   * r)

            # draw the predicted face name on the image
            green = (0, 255, 0) # pure green
            cv2.rectangle(frame, (left, top), (right, bottom),	green, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2)

        # if the video writer is None *AND* we are supposed to write the output video to disk initialize the writer
        if writer is None and output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 20,
                (frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized faces to disk
        if writer is not None:
            writer.write(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # cleanup
    cv2.destroyAllWindows()
    vs.stop()

    # check to see if the video writer point needs to be released
    if writer is not None:
        writer.release()


if __name__ == '__main__':
    main()
