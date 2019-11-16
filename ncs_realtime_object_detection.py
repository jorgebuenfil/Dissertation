# !/usr/bin/python3
'''
All of our data is stored in one long array/list ( output ).
Using the box_index, we calculate our base_index which we’ll then use
(with more offsets) to extract prediction data.
The output list has the following format:

    output[0]              : num_valid_boxes
    output[base_index + 1] : prediction class index
    output[base_index + 2] : prediction confidence
    output[base_index + 3] : object boxpoint x1 value (it needs to be scaled)
    output[base_index + 4] : object boxpoint y1 value (it needs to be scaled)
    output[base_index + 5] : object boxpoint x2 value (it needs to be scaled)
    output[base_index + 6] : object boxpoint y2 value (it needs to be scaled)
'''
from imutils.video import VideoStream
from imutils.video import FPS
import mvnc_multidevice_functions as mv            # made by Jorge
import numpy as np
import threading
import argparse
import time
import vlc
import cv2
import os
'''
Initialize the list of class labels our network was trained to detect, 
then generate a set of bounding box colors for each class.
Available classes:
    0: background
    1: aeroplane
    2: bicycle
    3: bird
    4: boat
    5: bottle
    6: bus
    7: car
    8: cat
    9: chair
    10: cow
    11: diningtable
    12: dog
    13: horse
    14: motorbike
    15: person
    16: pottedplant
    17: sheep
    18: sofa
    19: train
    20: tvmonitor
'''
CLASSES    = ("background", "aeroplane", "bicycle", "bird",
              "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS     = np.random.uniform(0, 255, size=(len(CLASSES), 3))

graph_file = "/home/jorge/workspace/ncappzoo/caffe/SSD_MobileNet/mobilenetgraph"
'''
Frame dimensions have to be square.
MobileNet SSD requires dimensions of 300×300, but we’ll be displaying the video stream at 
900×900 to better visualize the output
'''
PREPROCESS_DIMS  = (300, 300)
DISPLAY_DIMS     = (600, 600)
camera_source    = 1    # source = 0 is the default computer camera; 1 is a USB camera
confidence_level = 0.5  # 50%
DISP_MULTIPLIER  = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]  # multiplier to scale bounding boxes

class action_Thread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name     = name
        self.counter  = counter
        self.audio_warning = vlc.MediaPlayer("/home/jorge/Music/Bad_Boys.mp3")

    def run(self):
        audio_warning.play()
        time.sleep(1)
        audio_warning.stop()


def preprocess_image(input_image):
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    preprocessed = preprocessed - 127.5
    preprocessed = preprocessed * 0.007843
    preprocessed = preprocessed.astype(np.float32)
    return preprocessed

def predict(image, graph):
    # Initialize the list of predictions
    predictions = []

    image = preprocess_image(image)
    output, inference_time = mv.infer_image(graph, image)

    # Obtain the number of valid object predictions
    num_valid_boxes = output[0]
    print("valid boxes = {}".format(num_valid_boxes))

    # loop over results
    for box_index in range(int(num_valid_boxes)):
    # calculate the base index into our array so we can extract bounding box info
    # base_index = (box_index * 7) + 7
        base_index = 7 + box_index * 7

        # This ensures that we have valid data.
        if (not np.isfinite(output[base_index]) or
            not np.isfinite(output[base_index + 1]) or
            not np.isfinite(output[base_index + 2]) or
            not np.isfinite(output[base_index + 3]) or
            not np.isfinite(output[base_index + 4]) or
            not np.isfinite(output[base_index + 5]) or
            not np.isfinite(output[base_index + 6])):
            continue

        # extract the image width and height
        (h, w) = image.shape[:2]

        # Clip boxes to image size (in case network returns boxes outside of image boundaries.
        # Lower left corner
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))

        # Upper right corner
        x2 = min(w, int(output[base_index + 5] * w))
        y2 = min(h, int(output[base_index + 6] * h))

        # Get prediction class index, confidence level, and bounding box coordinates
        pred_class  = int(output[base_index + 1])
        pred_conf   = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))

        # Build prediction tuple
        prediction = (pred_class, pred_conf, pred_boxpts)

        # Append the prediction to the current predictions list
        predictions.append(prediction)

    return predictions

def main():
    '''
    Open a pointer to the video stream thread and allow the buffer to start to fill,
    then start the FPS counter.
    Note: For Raspberry Pi: vs = VideoStream(usePiCamera=True).start()
          For Ubuntu 18.04: vs = VideoStream(camera_source.start())
    '''
    vs = VideoStream(camera_source).start()

    time.sleep(1)

    # Loop over frames from the video file stream
    current_predicted_class = []
    fps                     = FPS().start()

    device = mv.open_ncs_device
    graph  = mv.load_graph(device, graph_file)
    theme_song = action_Thread(1, "Bad Boys", 1)
    theme_song.start()

    while True:
        try:
            # Get frame from the threaded video stream
            frame = vs.read()

            # Copy frame and resize it for display/video purposes
            image_for_result = frame.copy()
            image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

            # Image to be processed needs to be converted to FP32 to work with NCSDK2
            tensor = frame.astype(np.float32)

            # use the NCS to acquire predictions
            output, inference_time = mv.infer_image(graph, tensor)

            # loop over our predictions
            for (i, pred) in enumerate(output):
                # extract prediction data for readability
                (pred_class, pred_conf, pred_boxpts) = pred

                if pred_conf > confidence_level:
                    predicted_class = CLASSES[pred_class]

                    # print prediction to terminal
                    if predicted_class == 'dog':
                        theme_song.run()
                        #audio_warning.play()
                        #time.sleep(1)
                        #audio_warning.stop()

                    if predicted_class not in current_predicted_class:
                        current_predicted_class.append(predicted_class)
                        print("{:.1f}% {} detected".format(pred_conf * 100, predicted_class))

                    # build a label consisting of the predicted class and associated probability
                    label = "{}: {:.1f}%".format(predicted_class, pred_conf * 100)

                    # extract information from the prediction boxpoints
                    (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                    ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
                    ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
                    (startX, startY) = (ptA[0], ptA[1])
                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    # display the rectangle and label text
                    cv2.rectangle(image_for_result, ptA, ptB, COLORS[pred_class], 2)
                    cv2.putText(image_for_result, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)

            # display the frame to the screen
            cv2.imshow("Output", image_for_result)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                print("\nDetected classes: {}\n".format(current_predicted_class))
                break

            # update the FPS counter
            fps.update()

        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            print("\nDetected classes: {}".format(current_predicted_class))
            break

        # if there's a problem reading a frame, break gracefully
        except AttributeError:
            break

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

    # display FPS information
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # clean up the graph and device
    mv.close_all(device, graph)

if __name__ == '__main__':
    main()
