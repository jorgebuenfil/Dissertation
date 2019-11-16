#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from sys import argv
import numpy as np
import datetime
import queue
import time
import cv2
import sys
import os

from googlenet_processor import googlenet_processor
from tiny_yolo_processor import tiny_yolo_processor
from video_processor     import video_processor

# the networks compiled for NCS via ncsdk tools
GRAPH_FILE_DIRECTORY = "/home/jorge/workspace/ncappzoo/caffe/"
TINY_YOLO_GRAPH_FILE = GRAPH_FILE_DIRECTORY + "TinyYolo/yolo_tiny.graph"
GOOGLENET_GRAPH_FILE = GRAPH_FILE_DIRECTORY + "GoogLeNet/googlenet.graph"

resize_output_width  = 640
resize_output_height = 480
resize_output        = True

VIDEO_QUEUE_PUT_WAIT_MAX       = 4
VIDEO_QUEUE_FULL_SLEEP_SECONDS = 0.01

# for title bar of GUI window
cv_window_name = 'ACE Demo: real-time object detection - Q to quit'

VIDEO_QUEUE_SIZE     =  2  # original value =  2
GN_INPUT_QUEUE_SIZE  = 10  # original value = 10
GN_OUTPUT_QUEUE_SIZE = 10  # original value = 10
TY_OUTPUT_QUEUE_SIZE = 10  # original value = 10

# number of seconds to wait when putting or getting from queue's besides the video output queue.
QUEUE_WAIT_MAX       =  2

# input and output queue for the GoogLeNet processor.
gn_input_queue  = queue.Queue(GN_INPUT_QUEUE_SIZE)
gn_output_queue = queue.Queue(GN_OUTPUT_QUEUE_SIZE)
video_queue    = None
video_proc     = None
ty_proc        = None
gn_proc        = None
gn_proc_list   = []
gn_device_list = []

# if True will do GoogLeNet inferences for each object returned from
# tiny yolo, if False will only do the tiny yolo inferences
do_gn = False

# read video files from this directory
input_video_path     = "/home/jorge/Videos/ACE/"
pause_mode           = False
font_scale           = 0.55
'''
Tuning variables

If GoogLeNet returns a probablity less than this then just use the tiny yolo more general classification (i.e. 'bird')
'''
GN_PROBABILITY_MIN   = 0.5
'''
Only keep boxes with probabilities greater than this when doing the tiny yolo filtering.  
This is only an initial value, pressing the B/b keys will adjust up or down respectively
'''
TY_INITIAL_BOX_PROBABILITY_THRESHOLD = 0.13
'''
The intersection-over-union threshold to use when determining duplicates.
Objects/boxes found that are over this threshold will be considered the same object 
when filtering the Tiny Yolo output.  This is only an initial value.  
Pressing the I/i key will adjust up or down respectively.
'''
TY_INITIAL_MAX_IOU   = 0.15

def overlay_on_image(display_image, filtered_objects):
    """
    Displays a gui window with an image that contains boxes and lables for found objects.
    The source_image is the image on which the inference was run.
    :param display_image:
    :param filtered_objects: list of lists (as returned from filter_objects() and then added
    to by get_googlenet_classifications().
    Each of the inner lists represent one found object and contain:
        [0]:string that is yolo network classification ie 'bird'.
        [1]:float value for box center X pixel location within source image.
        [2]:float value for box center Y pixel location within source image.
        [3]:float value for box width in pixels within source image.
        [4]:float value for box height in pixels within source image.
        [5]:float value that is the probability for the yolo classification.
        [6]:int value that is the index of the googlenet classification.
        [7]:string value that is the googlenet classification string.
        [8]:float value that is the googlenet probability
     :return: Returns True if should go to next image or False if should not.
    """
    DISPLAY_BOX_WIDTH_PAD  = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    source_image_width  = display_image.shape[1]
    source_image_height = display_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        center_x    = int(filtered_objects[obj_index][1])
        center_y    = int(filtered_objects[obj_index][2])
        half_width  = int(filtered_objects[obj_index][3])//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + DISPLAY_BOX_HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left   = max(center_x - half_width, 0)
        box_top    = max(center_y - half_height, 0)
        box_right  = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        #draw the rectangle on the image.  This is hopefully around the object
        box_color     = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color       = (255, 255, 255)   # white text

        if (filtered_objects[obj_index][8] > GN_PROBABILITY_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        label_left = box_left
        label_top  = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right  = label_left + label_size[0]
        label_bottom = label_top  + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def get_googlenet_classifications(source_image, filtered_objects):
    """
    Executes googlenet inferences on all objects defined by filtered_objects.
    To run the inferences will crop an image out of source image based on the
    boxes defined in filtered_objects and use that as input for googlenet.
    :param source_image: original image on which the inference was run.
    The boxes defined by filtered_objects are rectangles within this
    image and will be used as input for googlenet.
    :param filtered_objects: [IN/OUT] upon input is a list of lists
    (as returned from filter_objects() each of the inner lists represent
    one found object and contain the following 6 values:
        - string that is network classification ie 'cat', or 'chair' etc.
        - float value for box center X pixel location within source image.
        - float value for box center Y pixel location within source image.
        - float value for box width in pixels within source image.
        - float value for box height in pixels within source image.
        - float value that is the probability for the network classification.
   upon output the following 3 values from the googlenet inference will
   be added to each inner list of filtered_objects:
        - int value that is the index of the googlenet classification.
        - string value that is the googlenet classification string.
        - float value that is the googlenet probability
    :return: None.
    """
    if (not do_gn):
        for obj_index in range(len(filtered_objects)):
            filtered_objects[obj_index] += (0, '', 0.0)
        return

    source_image_width  = source_image.shape[1]
    source_image_height = source_image.shape[0]

    """ pad the height and width of the image boxes by this 
    amount to make sure we get the whole object in the image that we pass to GoogLeNet."""
    WIDTH_PAD  = int(source_image_width * 0.08) #80 #20
    HEIGHT_PAD = int(source_image_height* 0.08) #80 #30

    """  Loop through each box and crop the image in that rectangle from the source image 
    and then use it as input for GoogLeNet we are basically gathering the googlenet results
    serially here on the main thread.  we could have used a separate thread but probably 
    wouldn't give much improvement since tiny yolo is already working on another thread."""
    for obj_index in range(len(filtered_objects)):
        center_x    = int(filtered_objects[obj_index][1])
        center_y    = int(filtered_objects[obj_index][2])
        half_width  = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left   = max(center_x - half_width, 0)
        box_top    = max(center_y - half_height, 0)
        box_right  = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        # get one image by clipping a box out of source image
        one_image = source_image[box_top:box_bottom, box_left:box_right]
        gn_input_queue.put(one_image, True, QUEUE_WAIT_MAX)

    for obj_index in range(len(filtered_objects)):
        result_list = gn_output_queue.get(True, QUEUE_WAIT_MAX)
        filtered_objects[obj_index] += result_list

    return

def get_googlenet_classifications_no_queue(gn_proc, source_image, filtered_objects):
    '''
    Executes googlenet inferences on all objects defined by filtered_objects.
    To run the inferences will crop an image out of source image based on the boxes defined
    in filtered_objects and use that as input for GoogLeNet.
    :param gn_proc: GoogLeNet graph object on which the inference should be executed.
    :param source_image: original image on which the inference was run.
    :param filtered_objects: [IN/OUT] upon input is a list of lists (as returned
    from filter_objects() each of the inner lists represent one found object and contain:
    - string that is network classification ie 'cat', or 'chair' etc.
    - float value for box center X pixel location within source image.
    - float value for box center Y pixel location within source image.
    - float value for box width in pixels within source image.
    - float value for box height in pixels within source image.
    - float value that is the probability for the network classification.

    upon output the following 3 values from the googlenet inference will be added to
    each inner list of filtered_objects:
    - int value that is the index of the googlenet classification.
    - string value that is the googlenet classification string.
    - float value that is the googlenet probability.
    :return: returns None
    '''
    if (not do_gn):
        for obj_index in range(len(filtered_objects)):
            filtered_objects[obj_index] += (0, '', 0.0)
        return

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD  = 20
    HEIGHT_PAD = 30

    source_image_width  = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # id for all sub images within the larger image
    image_id = datetime.datetime.now().timestamp()

    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    for obj_index in range(len(filtered_objects)):
        center_x    = int(filtered_objects[obj_index][1])
        center_y    = int(filtered_objects[obj_index][2])
        half_width  = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left    = max(center_x - half_width, 0)
        box_top     = max(center_y - half_height, 0)
        box_right   = min(center_x + half_width, source_image_width)
        box_bottom  = min(center_y + half_height, source_image_height)

        # get one image by clipping a box out of source image
        one_image = source_image[box_top:box_bottom, box_left:box_right]

        # Get a googlenet inference on that one image and add the information
        # to the filtered objects list
        filtered_objects[obj_index] += gn_proc.googlenet_inference(one_image, image_id)

    return

def do_unpause():
    '''
    Unpauses the processing of frames.
    If already in pause mode does nothing After unpausing the video processor will wait for at least
    one frame to be put in the video output queue or for a few seconds which ever comes first.
    :return: returns None
    '''
    global video_proc, video_queue, pause_mode

    print("Unpausing")
    if (not pause_mode):
        # already in pause mode
        return

    # reset our global pause mode flag to indicate no longer in pause mode
    pause_mode = False

    # tell the video processor to unpause itself
    # when it starts processing frames again the rest of the
    # program will start picking up the frames and processing them
    video_proc.unpause()

    # now wait until at least one frame has been processed by the
    # video processor. Or time out after a few tries.
    count = 0
    while (video_queue.empty() and count < 20):
        # video queue still empty, so short sleep then try again
        time.sleep(0.1)
        count += 1

def handle_keys(raw_key):
    '''
    handles key presses by adjusting global thresholds etc.
    :param raw_key: return value from cv2.waitkey
    :return: returns False if program should end, or True if should continue
    '''
    ascii_code = raw_key & 0xFF
    if (ascii_code == ord('q')) or (ascii_code == ord('Q')):
        return False
    elif (ascii_code == ord('B')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() + 0.05)
        print("New tiny yolo box probability threshold is " + str(ty_proc.get_box_probability_threshold()))
    elif (ascii_code == ord('b')):
        ty_proc.set_box_probability_threshold(ty_proc.get_box_probability_threshold() - 0.05)
        print("New tiny yolo box probability threshold  is " + str(ty_proc.get_box_probability_threshold()))
    elif (ascii_code == ord('G')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN + 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('g')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN - 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('I')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() + 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))
    elif (ascii_code == ord('i')):
        ty_proc.set_max_iou(ty_proc.get_max_iou() - 0.05)
        print("New tiny yolo max IOU is " + str(ty_proc.get_max_iou() ))
    elif (ascii_code == ord('T')):
        font_scale += 0.1
        print("New text scale is: " + str(font_scale))
    elif (ascii_code == ord('t')):
        font_scale -= 0.1
        print("New text scale is: " + str(font_scale))
    elif (ascii_code == ord('p')):
        # pause mode toggle
        if (not pause_mode):
            print("pausing")
            pause_mode = True
            video_proc.pause()

        else:
            do_unpause()
    elif (ascii_code == ord('2')):
        do_gn = not do_gn
        print("New do GoogLeNet value is " + str(do_gn))

    return True

def print_usage():
    "prints usage information"
    print('\nusage: ')
    print('python3 street_cam_threaded.py [help][googlenet=on|off][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - Prints this message')
    print('  resize_window - Resizes the GUI window to specified dimensions')
    print('                  must be formatted similar to resize_window=1280x720')
    print('                  default behavior is to use source video frame size')
    print('  googlenet - Sets initial state for googlenet processing')
    print('              must be formatted as googlenet=on or googlenet=off')
    print('              When on all tiny yolo objects will be passed to googlenet')
    print('              for further classification, when off only tiny yolo will be used')
    print('              Default behavior is off')
    print('')
    print('Example: ')
    print('python3 street_cam_threaded.py googlenet=on resize_window=1920x1080')

def show_message(title, message, width=600):
    message_window = tk.Tk()
    message_window.title(title)
    msg = tk.Message(message_window, text=message, width=width)
    msg.config(bg='lightgreen', font=('times', 12, 'italic'))
    msg.pack()

def print_info():
    """
    Displays information for the user when program starts.
    """
    title   = "Running street_cam_threaded"
    message = "Keys:\n\n \
    'Q'/'q' to Quit\n \
    'B'/'b' to inc/dec the Tiny Yolo box probability threshold\n \
    'I'/'i' to inc/dec the Tiny Yolo box intersection-over-union threshold \n \
    'G'/'g' to inc/dec the GoogLeNet probability threshold \n \
    'T'/'t' to inc/dec the Text size for the labels \n \
    '2'     to toggle GoogLeNet inferences \n \
    'p'     to pause/unpause"
    show_message(title, message)

def init_gn_lists(enumerated_devices):
    ''' Initializes the googlenet processors and devices.
        "enumerated_devices" is a list of NCS devices to use for googlenet processing.
        Each device in the list will be used for google net processing.
        "gn_proc_list" is a list that will be populated with initialized googlenet_processor instances
      which will each be initialized with the same input and output queues to process googlenet
      inferences for the program.
        "gn_device_list" is a list that will be populated with the opened NCS devices initialized for
      googlenet processing.
        The device at index N corresponds to the googlenet_processor at index N.
      These device will need to be closed via the ncapi return True if worked or False if error
    '''
    try:
        for one_device in enumerated_devices:
            gn_device = mvnc.Device(one_device)
            gn_device.open()
            gn_proc = googlenet_processor(GOOGLENET_GRAPH_FILE, gn_device, gn_input_queue, gn_output_queue,
                                      QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)
            gn_proc_list.insert(0,   gn_proc)
            gn_device_list.insert(0, gn_device)
    except:
        return False

    return True

def assign_ncs_devices():
    '''
    use the first NCS device for tiny YOLO processing, and the rest for GoogLeNet processing
    '''
    global ty_device

    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)
    devices   = mvnc.enumerate_devices()
    ty_device = mvnc.Device(devices[0])
    ty_device.open()

    if not init_gn_lists(devices[1:]):
        print('Error while initializing NCS devices for GoogLeNet')
    return True

def create_GUI_window():
    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10, 10)
    cv2.waitKey(1)

def main():
    '''
    This function is called from the entry point to do all the work.
    '''
    if not assign_ncs_devices():
        print("Problem assigning NCS Devices!")
        return 1

    # print keyboard mapping to console so user will know what can be adjusted.
    # print_info()
    create_GUI_window()

    # Creates the queue of video frame images which will be the output for the video processor
    video_queue = queue.Queue(VIDEO_QUEUE_SIZE)

    # Setup tiny_yolo_processor that reads from the video queue and writes to its own ty_output_queue
    ty_output_queue = queue.Queue(TY_OUTPUT_QUEUE_SIZE)
    ty_proc         = tiny_yolo_processor(TINY_YOLO_GRAPH_FILE, ty_device, video_queue, ty_output_queue,
                                          TY_INITIAL_BOX_PROBABILITY_THRESHOLD, TY_INITIAL_MAX_IOU,
                                          QUEUE_WAIT_MAX, QUEUE_WAIT_MAX)
    exit_app = False
    while True:
         # video processor that will put video frames images on the video_queue
         video_file = askopenfilename(initialdir="/home/jorge/Videos/", title="Select videofile",
                                                    filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
         video_proc = video_processor(video_queue,
                                      video_file,
                                      VIDEO_QUEUE_PUT_WAIT_MAX,
                                      VIDEO_QUEUE_FULL_SLEEP_SECONDS)
         for gn_proc in gn_proc_list:
             gn_proc.start_processing()

         video_proc.start_processing()
         ty_proc.start_processing()

         frame_count = 0
         start_time  = time.time()
         end_time    = start_time
         total_paused_time = end_time - start_time

         while True:
            try:
                (display_image, filtered_objs) = ty_output_queue.get(True, QUEUE_WAIT_MAX)
            except:
                print("couldn't grab a frame")
                pass

            get_googlenet_classifications(display_image, filtered_objs)

            prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
            if (prop_val < 0.0):
                end_time = time.time()
                ty_output_queue.task_done()
                exit_app = True
                break

            overlay_on_image(display_image, filtered_objs)

            if resize_output:
                display_image = cv2.resize(display_image, (resize_output_width, resize_output_height), cv2.INTER_LINEAR)
            # update the GUI window with new image
            cv2.imshow(cv_window_name, display_image)
            ty_output_queue.task_done()
            raw_key = cv2.waitKey(1)
            if (raw_key != -1):
                if handle_keys(raw_key) == False:
                    end_time = time.time()
                    exit_app = True
                    break
                if (pause_mode):
                    pause_start = time.time()
                    while (pause_mode):
                        raw_key = cv2.waitKey(1)
                        if (raw_key != -1):
                            if (handle_keys(raw_key) == False):
                                end_time = time.time()
                                do_unpause()
                                exit_app = True
                                print('user pressed Q during pause')
                                break
                    if (exit_app):
                        break;
                    pause_stop        = time.time()
                    total_paused_time = total_paused_time + (pause_stop - pause_start)

            frame_count += 1

            if (video_queue.empty()):
                end_time = time.time()
                print('video queue empty')
                break

         frames_per_second = frame_count / ((end_time - start_time) - total_paused_time)
         print('Frames per Second: ' + str(frames_per_second))

         video_proc.stop_processing()
         ty_proc.stop_processing()
         for gn_proc in gn_proc_list:
             gn_proc.stop_processing()
         video_proc.cleanup()

         if (exit_app):
             break

    # clean up tiny yolo
    ty_proc.cleanup()
    ty_device.close()
    ty_device.destroy()

    # Clean up googlenet
    for gn_index in range(0, len(gn_proc_list)):
        gn_proc_list[gn_index].cleanup()
        gn_device_list[gn_index].close()

if __name__ == "__main__":
    main()
