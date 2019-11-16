#! /usr/bin/env python3
# stream_ty_gn.py

from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import time
import vlc

# imported libraries created or significantly modified by Jorge
import mvnc_functions as mv

# the networks compiled for NCS via ncsdk tools
tiny_yolo_graph_file = "/home/jorge/workspace/ncappzoo/caffe/TinyYolo/yolo_tiny.graph"
googlenet_graph_file = "/home/jorge/workspace/ncappzoo/caffe/GoogLeNet/googlenet.graph"
MEAN_FILE_NAME       = "/home/jorge/workspace/ncappzoo/data/ilsvrc12/ilsvrc_2012_mean.npy"
gn_labels_file       = "/home/jorge/workspace/ncappzoo/data/ilsvrc12/synset_words.txt"

"""
Initialize the list of class labels our network was trained to detect, then generate a set of 
bounding box colors for each class.
"""
CLASSES    = ("background", "aeroplane", "bicycle", "bird",
              "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train", "tvmonitor")
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
# Specifies which camera to use.  If only one it will likely be index 0
CAMERA_INDEX = 1

# Tiny Yolo assumes input images are these dimensions.
TY_NETWORK_IMAGE_WIDTH  = 448
TY_NETWORK_IMAGE_HEIGHT = 448

# GoogLeNet assumes input images are these dimensions
GN_NETWORK_IMAGE_WIDTH  = 224
GN_NETWORK_IMAGE_HEIGHT = 224

"""
Labels to display along with boxes if googlenet classification is good these will be 
read in from the synset_words.txt file for ilsvrc12.
"""
gn_labels = [""]

# for title bar of GUI window
cv_window_name = "STYG (Stream Tiny YOLO and GoogLeNet) - Q to quit"

# Requested and actual camera dimensions
REQUEST_CAMERA_WIDTH  = 1024 #TY_NETWORK_IMAGE_WIDTH
REQUEST_CAMERA_HEIGHT = 768  #TY_NETWORK_IMAGE_HEIGHT
actual_camera_width   = 0
actual_camera_height  = 0

"""
 Tuning variables

Only keep boxes with probabilities greater than this when doing the tiny YOLO filtering.
"""
TY_BOX_PROBABILITY_THRESHOLD = 0.10  # 0.07

# if GoogLeNet returns a probablity less than this then just use the tiny YOLO more general classification ie 'bird'
GN_PROBABILITY_MIN = 0.5

"""
The intersection-over-union threshold to use when determining duplicates.
Objects/boxes found that are over this threshold will be considered the same 
object when filtering the Tiny YOLO output.
"""
TY_MAX_IOU = 0.35
# end of tuning variables

def filter_objects(inference_result, input_image_width, input_image_height):
    """
    Interpret the output from a single inference of TinyYolo (GetResult)
    and filter out objects/boxes with low probabilities.
    Output is the array of floats returned from the API GetResult but converted
    to float32 format.
    :param inference_result: array of floats returned from the API GetResult but converted
    to float32 format.
    :param input_image_width: width of the input image.
    :param input_image_height: height of the input image.
    :return: list of lists. each of the inner lists represent one found object and contain
    the following 6 values:
        string that is network classification ie 'cat', or 'chair' etc.
        float value for box center X pixel location within source image.
        float value for box center Y pixel location within source image.
        float value for box width in pixels within source image.
        float value for box height in pixels within source image.
        float value that is the probability for the network classification.
    """
    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    # the 20 classes this network was trained on
    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # which types of objects do we want to include. A zero means we do not want the class in that position.
    network_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1]

    num_classifications = len(network_classifications) # should be 20
    grid_size           =  7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell =  2 # the number of boxes returned for each grid cell

    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    """
    Classification_probabilities contains a probability for each classification for each 
    64x64 pixel square of the grid.  The source image contains 7x7 of these 64x64 pixel 
    squares and there are 20 possible classifications. 7x7x20 = 980.
    """
    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:, :, box_index, class_index] = \
                np.multiply(classification_probabilities[:, :, class_index], box_prob_scale_factor[:, :, box_index])

    probability_threshold_mask      = np.array(all_probabilities >= TY_BOX_PROBABILITY_THRESHOLD, dtype='bool')
    box_threshold_mask              = np.nonzero(probability_threshold_mask)
    boxes_above_threshold           = all_boxes[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities, axis=3)[box_threshold_mask[0], box_threshold_mask[1], box_threshold_mask[2]]
    probabilities_above_threshold   = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then sort the probabilities and classifications to match
    argsort                         = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold           = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold   = probabilities_above_threshold[argsort]

    # get mask for boxes that seem to be the same object
    duplicate_box_mask              = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold           = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold   = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],
                                            boxes_above_threshold[i][0], boxes_above_threshold[i][1],
                                            boxes_above_threshold[i][2], boxes_above_threshold[i][3],
                                            probabilities_above_threshold[i]])

    return classes_boxes_and_probs

def get_duplicate_box_mask(box_list):
    """
    Creates a mask to remove duplicate objects (boxes) and their related probabilities
    and classifications that should be considered the same object.  This is determined
    by how similar the boxes are based on the Intersection-Over-Union metric.
    box_list is as list of boxes (4 floats for centerX, centerY and Length and Width)
    """
    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > TY_MAX_IOU:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask

def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):
    """
    Converts the boxes in box list to pixel units.
    :param box_list: output from the box output from the tiny yolo network.
    :param image_width:
    :param image_height:
    :param grid_size: [grid_size x grid_size x 2 x 4].
    :return: Intersection Over Union.
    """
    # number of boxes per grid cell
    boxes_per_cell = 2

    """
    Setup some offset values to map boxes to pixels.
    Box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    """
    box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

    # adjust the box center
    box_list[:, :, :, 0]  += box_offset
    box_list[:, :, :, 1]  += np.transpose(box_offset,(1,0,2))
    box_list[:, :, :, 0:2] = box_list[:, :, :, 0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:, :, :, 2] = np.multiply(box_list[:, :, :, 2],box_list[:, :, :, 2])
    box_list[:, :, :, 3] = np.multiply(box_list[:, :, :, 3], box_list[:, :, :, 3])

    #scale the boxes to the image size in pixels
    box_list[:, :, :, 0] *= image_width
    box_list[:, :, :, 1] *= image_height
    box_list[:, :, :, 2] *= image_width
    box_list[:, :, :, 3] *= image_height

def get_intersection_over_union(box_1, box_2):
    """
    Evaluate the intersection-over-union for two boxes.
    The intersection-over-union metric determines how close two boxes are to being the same box.
    The closer the boxes are to being the same, the closer the metric will be to 1.0

    :param box_1: array of 4 numbers which are the (x, y) points that define the center of the box
    and the length and width of the box.
    :param box_2: array of 4 numbers which are the (x, y) points that define the center of the box
    and the length and width of the box.
    :return: intersection-over-union (between 0.0 and 1.0) for the two boxes specified.
    """
    # one diminsion of the intersecting box
    intersection_dim_1 = min(box_1[0]+0.5*box_1[2], box_2[0]+0.5*box_2[2]) - \
                         max(box_1[0]-0.5*box_1[2], box_2[0]-0.5*box_2[2])

    # the other dimension of the intersecting box
    intersection_dim_2 = min(box_1[1]+0.5*box_1[3], box_2[1]+0.5*box_2[3]) - \
                         max(box_1[1]-0.5*box_1[3], box_2[1]-0.5*box_2[3])

    if intersection_dim_1 < 0 or intersection_dim_2 < 0:
        intersection_area = 0  # no intersection area
    else:
        # intersection area is product of intersection dimensions
        intersection_area = intersection_dim_1*intersection_dim_2

    """
    Calculate the union area which is the area of each box added and then we need to 
    subtract out the intersection area since it is counted twice (by definition it is in each box)
    """
    union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area

    # now we can return the intersection over union
    iou = intersection_area / union_area

    return iou

def overlay_on_image(display_image, filtered_objects):
    """
    Displays a gui window with an image that contains boxes and lables for found objects.

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
    :return: True if should go to next image or False if
# should not.
    """
    DISPLAY_BOX_WIDTH_PAD  =  0
    DISPLAY_BOX_HEIGHT_PAD = 20

    # copy image so we can draw on it.
    source_image_width  = display_image.shape[1]
    source_image_height = display_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        center_x    = int(filtered_objects[obj_index][1])
        center_y    = int(filtered_objects[obj_index][2])
        half_width  = int(filtered_objects[obj_index][3])//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + DISPLAY_BOX_HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left    = max(center_x - half_width, 0)
        box_top     = max(center_y - half_height, 0)
        box_right   = min(center_x + half_width, source_image_width)
        box_bottom  = min(center_y + half_height, source_image_height)

        '''
        draw the rectangle on the image (arguments are: image, top left corner, bottom-right corner, 
        color and thickness.
        '''
        box_thickness = 2
        box_color     = box_colors[CLASSES.index(filtered_objects[obj_index][0])]
        box_color     = [int(c) for c in box_color]

        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (198, 198, 198)   # greyish green background for text
        label_text_color       = (255, 255,   0)   # white text

        if (filtered_objects[obj_index][8] > GN_PROBABILITY_MIN):
            #label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
            label_text = filtered_objects[obj_index][7] + " " + \
                         str(int(filtered_objects[obj_index][8]*100)) + "%"
        else:
            label_text = filtered_objects[obj_index][0] + " " + \
                         str(int(filtered_objects[obj_index][5]*100)) + "%"

        if label_text[:5] == 'teddy':
            print("I see a doggie!")
            audio_warning.play()
            time.sleep(5)
            audio_warning.stop()

        label_size   = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left   = box_left
        label_top    = box_top - label_size[1]
        label_right  = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left-1, label_top-1), (label_right+1, label_bottom+1),
                      label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, label_text_color, 1)

    # display text to let user know how to quit (colors are BGR)
    cv2.rectangle(display_image, (0, 0), (100, 15), (0, 198, 0), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)

def get_googlenet_classifications(gn_graph, source_image, filtered_objects):
    """
    Executes GoogLeNet inferences on all objects defined by filtered_objects.
    To run the inferences will crop an image out of source image based on the boxes defined in
    filtered_objects and use that as input for GoogLeNet.

    :param gn_graph: GoogLeNet graph object on which the inference should be executed.
    :param source_image: original image on which the inference was run.
    :param filtered_objects: The boxes defined by filtered_objects are rectangles
    within this image and will be used as input for GoogLeNet.
    Filtered_objects [IN/OUT] upon input is a list of lists (as returned from
    filter_objects() each of the inner lists represent one found object and contain:
        - string that is network classification ie 'cat', or 'chair' etc.
        - float value for box center X pixel location within source image.
        - float value for box center Y pixel location within source image.
        - float value for box width in pixels within source image.
        - float value for box height in pixels within source image.
        - float value that is the probability for the network classification.

    Upon output the following 3 values from the GoogLeNet inference will be added to
    each inner list of filtered_objects:
        - int value that is the index of the GoogLeNet classification.
        - string value that is the GoogLeNet classification string.
        - float value that is the GoogLeNet probability
    :return: None
    """
    """
    Pad the height and width of the image boxes by this amount to make sure we get the 
    whole object in the image that we pass to GoogLeNet.
    """
    WIDTH_PAD  = 20
    HEIGHT_PAD = 30

    source_image_width  = source_image.shape[1]
    source_image_height = source_image.shape[0]

    """
    Loop through each box and crop the image in that rectangle from the source image and 
    then use it as input for GoogLeNet.
    """
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

        # Get a GoogLeNet inference on that one image and add the information to the filtered objects list
        filtered_objects[obj_index] += googlenet_inference(gn_graph, one_image)

    return

def googlenet_inference(gn_graph, input_image):
    """
    Executes an inference using the GoogLeNet graph and image passed gn_graph is the
    GoogLeNet graph object to use for the inference.  It is assumed that this has been
    created with allocate graph and the GoogLeNet graph file on an open NCS device.

    :param gn_graph: GoogLeNet graph file on an open NCS device.
    :param input_image: image on which a GoogLeNet inference should be executed.
    :return: list of the following three items:
        - index of the most likely classification from the inference.
        - label for the most likely classification from the inference.
        - probability the most likely classification from the inference.
    """
    """
    Resize image to GoogLeNet network width and height then convert to float32, normalize (divide by 255), 
    and pass to LoadTensor as input for an inference
    """
    input_image = cv2.resize(input_image, (GN_NETWORK_IMAGE_WIDTH, GN_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
    input_image = input_image.astype(np.float32)
    input_image[:, :, 0] = (input_image[:, :, 0] - gn_mean[0])
    input_image[:, :, 1] = (input_image[:, :, 1] - gn_mean[1])
    input_image[:, :, 2] = (input_image[:, :, 2] - gn_mean[2])

    # Load tensor and get result.  This executes the inference on the NCS
    gn_graph.queue_inference_with_fifo_elem(gn_input_fifo, gn_output_fifo, input_image, None)
    output, userobj = gn_output_fifo.read_elem()

    order = output.argsort()[::-1][:1]

    '''
    print('\n------- prediction --------')
    for i in range(0, 1):
        print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[
            order[i]] + '  label index is: ' + str(order[i]))
    '''
    # index, label, probability
    return order[0], gn_labels[order[0]], output[order[0]]

def handle_keys(raw_key):
    """
    Handles key presses by adjusting global thresholds etc.
    :param raw_key: return value from cv2.waitkey.
    :return: False if program should end, or True if should continue.
    """
    global GN_PROBABILITY_MIN, TY_MAX_IOU, TY_BOX_PROBABILITY_THRESHOLD

    ascii_code = raw_key & 0xFF

    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        TY_BOX_PROBABILITY_THRESHOLD = TY_BOX_PROBABILITY_THRESHOLD + 0.05
        print("New TY_BOX_PROBABILITY_THRESHOLD is " + str(TY_BOX_PROBABILITY_THRESHOLD))
    elif (ascii_code == ord('b')):
        TY_BOX_PROBABILITY_THRESHOLD = TY_BOX_PROBABILITY_THRESHOLD - 0.05
        print("New TY_BOX_PROBABILITY_THRESHOLD is " + str(TY_BOX_PROBABILITY_THRESHOLD))
    elif (ascii_code == ord('G')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN + 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('g')):
        GN_PROBABILITY_MIN = GN_PROBABILITY_MIN - 0.05
        print("New GN_PROBABILITY_MIN is " + str(GN_PROBABILITY_MIN))
    elif (ascii_code == ord('I')):
        TY_MAX_IOU = TY_MAX_IOU + 0.05
        print("New TY_MAX_IOU is " + str(TY_MAX_IOU))
    elif (ascii_code == ord('i')):
        TY_MAX_IOU = TY_MAX_IOU - 0.05
        print("New TY_MAX_IOU is " + str(TY_MAX_IOU))
    return True

def print_info():
    """
    Prints information for the user when program starts.
    :return: none.
    """
    print('Running stream_ty_gn')
    print('Keys:')
    print("  'Q'/'q' to Quit")
    print("  'B'/'b' to inc/dec the Tiny Yolo box probability threshold")
    print("  'I'/'i' to inc/dec the Tiny Yolo box intersection-over-union threshold")
    print("  'G'/'g' to inc/dec the GoogLeNet probability threshold")
    print('')

def preprocess_image(input_image):
    """
    Resize image to network width and height then convert to float32, normalize (divide by 255),
    and pass to LoadTensor as input for inference.
    """
    input_image = cv2.resize(input_image, (TY_NETWORK_IMAGE_WIDTH, TY_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
    # save a display image as read from camera.
    display_image = input_image.copy()

    # modify input_image for TinyYolo input
    input_image = input_image.astype(np.float32)
    input_image = np.divide(input_image, 255.0)
    input_image = input_image[:, :, ::-1]  # convert to RGB

    return input_image, display_image

def assign_ncs_devices():
    '''
    use the first NCS device for tiny YOLO processing, and the rest for GoogLeNet processing
    '''
    global ty_device, gn_device

    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)
    devices   = mvnc.enumerate_devices()

    if len(devices) < 2:
        print('This application requires two NCS devices.')
        print('Insert two devices and try again!')
        return 1

    ty_device = mvnc.Device(devices[0])
    ty_device.open()

    gn_device = mvnc.Device(devices[1])
    gn_device.open()

def main():
    global gn_mean, gn_labels, actual_camera_height, actual_camera_width, \
        ty_graph, gn_graph, ty_input_fifo, ty_output_fifo, gn_input_fifo, gn_output_fifo, \
        box_colors, audio_warning

    box_colors    = np.random.randint(0, 256, size=(len(CLASSES), 3))

    audio_warning = vlc.MediaPlayer("/home/jorge/Music/Bad_Boys.mp3")
    print_info()
    assign_ncs_devices()

    #Load tiny yolo graph from disk and allocate graph via API
    try:
        with open(tiny_yolo_graph_file, mode='rb') as ty_file:
            ty_graph_from_disk        = ty_file.read()
        ty_graph                      = mvnc.Graph(tiny_yolo_graph_file)
        ty_input_fifo, ty_output_fifo = ty_graph.allocate_with_fifos(ty_device, ty_graph_from_disk)
    except:
        print ('Error - could not load tiny yolo graph file')
        mv.close_only_ncs_device(ty_device)
        mv.close_only_ncs_device(gn_device)
        return 1

    #Load GoogLeNet graph from disk and allocate graph via API
    try:
        with open(googlenet_graph_file, mode='rb') as gn_file:
            gn_graph_from_disk = gn_file.read()
        gn_graph                      = mvnc.Graph(googlenet_graph_file)
        gn_input_fifo, gn_output_fifo = gn_graph.allocate_with_fifos(gn_device, gn_graph_from_disk)
    except:
        print ('Error - could not load GoogLeNet graph file')
        mv.close_only_ncs_device(ty_device)
        mv.close_only_ncs_device(gn_device)
        return 1

    # GoogLenet initialization
    gn_mean      = np.load(MEAN_FILE_NAME).mean(1).mean(1)
    gn_labels    = np.loadtxt(gn_labels_file, str, delimiter='\t')

    for label_index in range(0, len(gn_labels)):
        temp = gn_labels[label_index].split(',')[0].split(' ', 1)[1]
        gn_labels[label_index] = temp

    print('Starting GUI, press Q to quit')
    #initialize_camera()
    cv2.namedWindow(cv_window_name)
    cv2.waitKey(1)

    camera_device = cv2.VideoCapture(CAMERA_INDEX)  # Camera instantiation
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width  = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if ((camera_device == None) or (not camera_device.isOpened())):
        print('Could not open camera.  Make sure it is plugged in.')
        print('Also, if you installed python opencv via pip or pip3 you')
        print('need to uninstall it and install from source with -D WITH_V4L=ON')
        print('Use the provided script: install-opencv-from_source.sh')
    frame_count = 0
    start_time  = time.time()
    end_time    = time.time()

    while True:
        # Read image from camera,
        ret_val, input_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, I'm out!")
            break

        input_image, display_image = preprocess_image(input_image)

        # Load tensor and get result.  This executes the inference on the NCS.
        ty_graph.queue_inference_with_fifo_elem(ty_input_fifo, ty_output_fifo, input_image, None)
        output, userobj = ty_output_fifo.read_elem()

        # filter out all the objects/boxes that don't meet thresholds
        filtered_objs = filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0])
        get_googlenet_classifications(gn_graph, display_image, filtered_objs)

        # check if the window is visible, this means the user hasn't closed the window via the X button.
        prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            end_time = time.time()
            break

        overlay_on_image(display_image, filtered_objs)
        """
        Resize back to original camera size so image doesn't look squashed. 
        It might be better to resize the boxes to match camera dimensions and overlay them 
        directly on the camera size image.
        """
        display_image = cv2.resize(display_image, (int(actual_camera_width), int(actual_camera_height)),
                                   cv2.INTER_LINEAR)
        # update the GUI window with new image
        cv2.imshow(cv_window_name, display_image)

        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                end_time = time.time()
                break

        frame_count = frame_count + 1

    frames_per_second = frame_count / (end_time - start_time)
    print ('Frames per Second: ' + str(frames_per_second))

    # close camera
    camera_device.release()

    # clean up devices
    ty_input_fifo.destroy()
    ty_output_fifo.destroy()
    ty_graph.destroy()
    ty_device.close()
    ty_device.destroy()
    gn_input_fifo.destroy()
    gn_output_fifo.destroy()
    gn_graph.destroy()
    gn_device.close()
    gn_device.destroy()

if __name__ == "__main__":
    sys.exit(main())
