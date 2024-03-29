#! /usr/bin/env python3

import sys
import numpy
import cv2
import time
import csv
import os
import sys
from sys import argv

# imported libraries created or significantly modified by Jorge
import mvnc_functions as mv

# name of the opencv window
cv_window_name = "SSD Mobilenet"

# labels AKA classes.  The class IDs returned are the indices into this list
labels = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH  = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
min_score_percent = 60

# the resize_window arg will modify these if its specified on the commandline
resize_output        = False
resize_output_width  = 0
resize_output_height = 0

# read video files from this directory
input_video_path = "/home/jorge/Videos/ACE/"

def preprocess_image(source_image):
    """
    Creates a preprocessed image from the source image that complies to the network expectations and returns it.
    :param source_image:
    :return: preprocessed image from the source image that complies to the network expectations.
    """
    resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    
    # trasnform values from range 0-255 to range -1.0 - 1.0
    resized_image = resized_image - 127.5
    resized_image = resized_image * 0.007843
    return resized_image

def handle_keys(raw_key):
    """
    Handles key presses by adjusting global thresholds etc.
    :param raw_key: raw_key is the return value from cv2.waitkey
    :return: False if program should end, or True if should continue
    """
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent += 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent -= 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True

def overlay_on_image(display_image, object_info):
    """
    Overlays the boxes and labels onto the display image.
    :param display_image: image on which to overlay the boxes/labels.
    :param object_info: list of 7 values as returned from the network.
          These 7 values describe the object found and they are:
          0: image_id (always 0 for myriad)
          1: class_id (this is an index into labels)
          2: score (this is the probability for the class)
          3: box left location within image as number between 0.0 and 1.0
          4: box top location within image as number between 0.0 and 1.0
          5: box right location within image as number between 0.0 and 1.0
          6: box bottom location within image as number between 0.0 and 1.0
    :return: None.
    """
    source_image_width  = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id   = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        return

    label_text = labels[int(class_id)] + " (" + str(percentage) + "%)"
    box_left   = int(object_info[base_index + 3] * source_image_width)
    box_top    = int(object_info[base_index + 4] * source_image_height)
    box_right  = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color     = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - min_score_percent)
    scaled_prob = (percentage - min_score_percent)
    scale = scaled_prob / scale_max

    # draw the classification label string just above and to the left of the rectangle
    # label_background_color = (70, 120, 70)  # greyish green background for text
    label_background_color = (0, int(scale * 175), 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top  = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right  = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def handle_args():
    """
    :return: False if found invalid args or True if processed ok
    """
    global resize_output, resize_output_width, resize_output_height
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).startswith('resize_window=')):
            try:
                arg, val     = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width  = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False
    return True

def run_inference(self, image_to_classify, ssd_mobilenet_graph):
    """
    Run an inference on the passed image.
    :param self:
    :param image_to_classify: image on which an inference will be performed. Upon successful
    return this image will be overlayed with boxes and labels identifying the found objects within the image.
    :param ssd_mobilenet_graph: Graph object from the NCAPI which will be used to peform the inference.
    :return: None.
    """
    global output, inference_time

    # preprocess the image to meet nework expectations
    resized_image = preprocess_image(image_to_classify)
    resized_image = resized_image.astype(numpy.float32)

    # Send the image to the NCS as 32 bit floats
    output, inference_time = infer_image(ssd_mobilenet_graph, resized_image)

    """
    a.	First fp32 value holds the number of valid detections = num_valid.
    b.	The next 6 values are unused.
    c.	The next (7 * num_valid) values contain the valid detections data
        Each group of 7 values will describe an object/box These 7 values in order.
        The values are:
            0: image_id (always 0)
            1: class_id (this is an index into labels)
            2: score (this is the probability for the class)
            3: box left location within image as number between 0.0 and 1.0
            4: box top location within image as number between 0.0 and 1.0
            5: box right location within image as number between 0.0 and 1.0
            6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    """
    num_valid_boxes = int(output[0])

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            x1 = max(int(output[base_index + 3] * image_to_classify.shape[0]), 0)
            y1 = max(int(output[base_index + 4] * image_to_classify.shape[1]), 0)
            x2 = min(int(output[base_index + 5] * image_to_classify.shape[0]), image_to_classify.shape[0]-1)
            y2 = min((output[base_index + 6] * image_to_classify.shape[1]), image_to_classify.shape[1]-1)

            # overlay boxes and labels on to the image
            overlay_on_image(image_to_classify, output[base_index:base_index + 7])

def print_usage():
    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080')

def main():
    global resize_output, resize_output_width, resize_output_height

    if (not handle_args()):
        print_usage()
        return 1

    graph_file = 'graph'

    # Open the NCS
    device              = mv.open_ncs_device()
    ssd_mobilenet_graph = mv.load_graph(device, graph_file)

    # get list of all the .mp4 files in the image directory
    input_video_filename_list = os.listdir(input_video_path)
    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]

    if (len(input_video_filename_list) < 1):
        # no videos to show
        print('No video (.mp4) files found')
        return 1

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)

    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list:

            cap = cv2.VideoCapture(input_video_file)

            actual_frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

            if ((cap == None) or (not cap.isOpened())):
                print ('Could not open video device.  Make sure file exists:')
                print ('file name:' + input_video_file)
                print ('Also, if you installed python opencv via pip or pip3 you')
                print ('need to uninstall it and install from source with -D WITH_V4L=ON')
                print ('Use the provided script: install-opencv-from_source.sh')
                exit_app = True
                break

            frame_count = 0
            start_time  = time.time()
            end_time    = start_time

            while(True):
                ret, display_image = cap.read()

                if (not ret):
                    end_time = time.time()
                    print("No image from from video device, exiting")
                    break

                # check if the window is visible, this means the user hasn't closed
                # the window via the X button
                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    exit_app = True
                    break

                run_inference(display_image, ssd_mobilenet_graph)

                if (resize_output):
                    display_image = cv2.resize(display_image,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)
                cv2.imshow(cv_window_name, display_image)

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key) == False):
                        end_time = time.time()
                        exit_app = True
                        break
                frame_count += 1

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            cap.release()

            if (exit_app):
                break;

        if (exit_app):
            break
    # Clean up the graph and the device
    mv.close_ncs_device(device, ssd_mobilenet_graph)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())

