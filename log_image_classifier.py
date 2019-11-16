#!/usr/bin/python3
# How to sequentially classify multiple images using DNNs on
# Intel® Movidius™ Neural Compute Stick (NCS)

from pathlib import Path
from skimage import io, transform
import tkinter as tk
from tkinter import filedialog
import mvnc_functions as mv
from glob import glob
import skimage
import numpy
import sys
import csv
import os

# User modifiable input parameters
IMAGE_EXTENSION       = "*.jpeg"
IMAGE_EXTENSION2      = "*.jpg"
default_image_folder  = "/home/jorge/Pictures/"
BASE_DIRECTORY        = "/home/jorge/workspace/ncappzoo/"
graph_file            = BASE_DIRECTORY + "caffe/GoogLeNet/graph"
LABELS_PATH           = BASE_DIRECTORY + "data/ilsvrc12/synset_words.txt"
IMAGE_MEAN            = numpy.float32([104.00698793, 116.66876762, 122.67891434])
MAX_IMAGE_COUNT       = 100
IMAGE_STDDEV          = (1)
IMAGE_DIM             = (224, 224)


def show_message(title, message, width=600):
    message_window = tk.Tk()
    message_window.title(title)
    msg = tk.Message(message_window, text=message, width=width)
    msg.config(bg='lightgreen', font=('times', 12, 'italic'))
    msg.pack()
    message_window.mainloop()

def compile_image_list():
    image_folder = filedialog.askdirectory(title     ="Choose a directory",
                                           initialdir=os.path.dirname("/home/jorge/Pictures/"),
                                           mustexist =True)
    if not image_folder:
        image_folder = default_image_folder

    file_list = [y for x in os.walk(image_folder)
                  for y in (glob(os.path.join(x[0], IMAGE_EXTENSION)))]
    print("got {} elements  for extension {}".format(len(file_list), IMAGE_EXTENSION))

    file_list_tmp = [y for x in os.walk(image_folder)
                  for y in (glob(os.path.join(x[0], IMAGE_EXTENSION2)))]
    file_list += file_list_tmp
    print("got {} elements  for extension {}".format(len(file_list_tmp), IMAGE_EXTENSION2))

    return file_list

def pre_process_image():
    # Read all images in the folder
    img_array       = []
    print_img_array = []

    file_list = compile_image_list()

    for file_index, file_name in enumerate(file_list):
        # Set a limit on the image count, so that it doesn't fill up the memory
        if file_index >= MAX_IMAGE_COUNT:
            break

        # Read & resize image [Image size is defined during training]
        #print("Processing {}".format(file_name))
        img = skimage.io.imread(file_name)
        print_img_array.append(skimage.transform.resize(img, (700, 700)))
        img = skimage.transform.resize(img, IMAGE_DIM, preserve_range=True)

        # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
        img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = img.astype(numpy.float32)
        img = (img - IMAGE_MEAN) * IMAGE_STDDEV

        img_array.append(img)

    return file_list, img_array, print_img_array

def infer_image():
    global results
    # Load the labels file
    labels =[line.rstrip('\n') for line in
                   open(LABELS_PATH) if line != 'top_predictions\n']

    # prepare the list of images from directory
    file_list, imgarray, print_imgarray = pre_process_image()

    device = mv.open_ncs_device()
    graph  = mv.load_graph(device, graph_file)
    for index, img in enumerate(imgarray):
        output, inference_time = mv.infer_image(graph, img)

        # Determine index of top 5 categories
        top_predictions = output.argsort()[::-1][:5]

        with open('jorges_inferences.csv', 'a', newline  ='\n') as csvfile:
            inference_log = csv.writer(csvfile, delimiter='@',
                                       quotechar='|',
                                       quoting  =csv.QUOTE_MINIMAL)

            inference_log.writerow([file_list[index],
                                   labels[top_predictions[0]], output[top_predictions[0]],
                                   labels[top_predictions[1]], output[top_predictions[1]],
                                   labels[top_predictions[2]], output[top_predictions[2]],
                                   labels[top_predictions[3]], output[top_predictions[3]],
                                   labels[top_predictions[4]], output[top_predictions[4]],
                                   numpy.sum(inference_time)])
    mv.close_ncs_device(device, graph)
    results = "Inference complete! View results in './jorges_inferences.csv'."
    return results

def main():
    infer_image()
    show_message("Results", results)

if __name__ == '__main__':
    sys.exit(main())
