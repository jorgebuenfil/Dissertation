#!/usr/bin/python3
'''
DIY smart security camera PoC using Intel® Movidius™ Neural Compute Stick."

Detect objects on a LIVE camera feed using Intel® Movidius™ Neural Compute Stick (NCS)
'''
import os
import cv2
import numpy
import argparse

#import mvnc.mvncapi as mvnc
import mvnc_functions as mv # made by Jorge

from time import localtime, strftime
from utils import visualize_output
from utils import deserialize_output

graph_file  = "/home/jorge/workspace/ncappzoo/caffe/SSD_MobileNet/graph"
labels_file = "/home/jorge/workspace/ncappzoo/caffe/SSD_MobileNet/labels.txt"
dim         = (300, 300)
colormode   = "bgr"
mean        = [127.5, 127.5, 127.5]
scale       = 0.00789
video       = 1

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDENCE_THRESHOLD = 0.60 # 60% confidant

# "Class of interest" - Display detections only if they match this class ID
BANNED_CLASS= 5 # Bottle
'''
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

# OpenCV object for video capture
camera               = None

def pre_process_image( frame ):
    # Resize image [Image size is defined by choosen network, during training]
    img = cv2.resize(frame, dim)

    # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
    if(colormode == "rgb"):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(numpy.float32)
    img = (img - numpy.float32(mean)) * scale

    return img

def infer_image(graph, img, frame):
    # Get the results from NCS
    output, inference_time = mv.infer_image(graph, img)

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd(output, CONFIDENCE_THRESHOLD, frame.shape )

    # Print the results (each image/frame may have multiple objects)
    for i in range(0, output_dict['num_detections']):
        # Filter a specific class/category
        if (output_dict.get('detection_classes_' + str(i) ) == BANNED_CLASS):
            print("Illegal object found!")
        cur_time = strftime( "%Y_%m_%d_%H_%M_%S", localtime() )

        # Extract top-left & bottom-right coordinates of detected objects
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

        # Prep string to overlay on the image
        display_str = (
            labels[output_dict.get('detection_classes_' + str(i))]
            + ": "
            + str( output_dict.get('detection_scores_' + str(i) ) )
            + "%" )

        # Overlay bounding boxes, detection class and scores
        frame = visualize_output.draw_bounding_box(
                    y1, x1, y2, x2,
                    frame,
                    thickness   = 4,
                    color       = (255, 255, 0),
                    display_str = display_str )

        # Capture snapshots
        photo = (os.path.dirname(os.path.realpath(__file__))
                  + "/captures/photo_"
                  + cur_time + ".jpg" )
        cv2.imwrite(photo, frame)

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS live inference', frame )

# ---- Main function (entry point for this script ) --------------------------
def main():
    global labels

    # Create a VideoCapture object
    camera = cv2.VideoCapture(video)

    # Set camera resolution
    camera.set( cv2.CAP_PROP_FRAME_WIDTH,  620 )
    camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )

    # Load the labels file
    labels     =[ line.rstrip('\n') for line in
              open(labels_file) if line != 'classes\n']
    ncs_device = mv.open_ncs_device()
    graph      = mv.load_graph(ncs_device, graph_file)

    # Main loop: Capture live stream & send frames to NCS
    while( True ):
        ret, frame = camera.read()
        img = pre_process_image(frame)
        infer_image(graph, img, frame)

        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
        if (cv2.waitKey(5) & 0xFF == ord('q')):
            break
    mv.close_ncs_device(ncs_device, graph)
    camera.release()
    cv2.destroyAllWindows()

# ---- Define 'main' function as the entry point for this script -------------
if __name__ == '__main__':
    main()
