#!/usr/bin/python3
# ****************************************************************************
# Author: Jorge Buenfil
# Single Shot Multibox Detectors (SSD)
# using the Intel® Movidius™ Neural Compute Stick (NCS)
# ****************************************************************************
import numpy
import skimage.io
import skimage.transform
import mvnc_functions as mv # made by Jorge
from utils import visualize_output
from utils import deserialize_output

CONFIDENCE_THRESHOLD = 0.60 # 60% confident
BASE_FILE   = "/home/jorge/workspace/ncappzoo/caffe/SSD_MobileNet/"
graph_file  = BASE_FILE + "graph"
labels_file = BASE_FILE + "labels.txt"

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
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
COLORS     = numpy.random.uniform(0, 255, size=(len(CLASSES), 3))

labels = [line.rstrip('\n') for line in
          open(labels_file) if line != 'classes\n']
colormode   = "bgr"
dim         = (300, 300)
scale       = 0.00789
mean        = [127.5, 127.5, 127.5]
test_image  = "/home/jorge/Pictures/dog.jpg"

def show_message(title, message):
    message_window = tk.Tk()
    message_window.title(title)
    msg = tk.Message(message_window, text = message, width = 600)
    msg.config(bg = 'lightblue', font = ('times', 12, 'italic'))
    msg.pack()
    tk.mainloop()

def ssd_pre_process_image(image_name):
    input_image    = skimage.io.imread(image_name)
    original_image = input_image

    # Resize image to size defined during training
    tensor = skimage.transform.resize(input_image, dim, preserve_range=True)

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    if (colormode == "bgr"):
        tensor = tensor[:, :, ::-1]

    # Mean subtraction & scaling
    tensor = tensor.astype(numpy.float32)
    tensor = (tensor - numpy.float32(mean)) * scale

    return tensor

def ssd_infer_image(tensor, image_name):
    original_image         = skimage.io.imread(image_name)
    output, inference_time = mv.manage_NCS(graph_file, tensor)
    output_dict            = deserialize_output.ssd(output, CONFIDENCE_THRESHOLD, original_image.shape)

    # Compile results
    results = "SSD Object Detection results:\n\n This image contains:\n"

    for i in range(0, output_dict['num_detections']):
        results = results + str(output_dict['detection_scores_' + str(i)]) + "% confidence it could be a " + \
                  labels[int(output_dict['detection_classes_'   + str(i)])][3:] + "\n"

        # Draw bounding boxes around valid detections
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

        # Prep string to overlay on the image
        display_str = (labels[output_dict.get('detection_classes_' + str(i))]
                       + str(output_dict.get('detection_scores_' + str(i))) + "%")

        original_image = visualize_output.draw_bounding_box(y1, x1, y2, x2,
                                                            original_image,
                                                            thickness  =6,
                                                            color      =(255, 255, 0),
                                                            display_str=display_str)
    results = results + "\n\nExecution time: " + str(int(numpy.sum(inference_time))) + " ms"

    return original_image, results

def main(image_name):
    tensor = ssd_pre_process_image(image_name)
    return ssd_infer_image(tensor, image_name)

if __name__ == '__main__':
    main(test_image)
