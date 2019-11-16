#!/usr/bin/python3
'''
Sequentially classify multiple images using GoogLeNet running on Intel® Movidius™ Neural Compute Stick (NCS).
Future planned enhancement: Add a selection menu for which file extensions to use (jpg, gif, png, jpeg, etc.)
'''
import os
import sys
import glob
import numpy
import ntpath
import argparse
import skimage.io
import skimage.transform
from tkinter import filedialog

# imported libraries created or significantly modified by Jorge
import utils.mvnc_functions as mv # made by Jorge

# Max number of images to process. 
# Avoid exhausting the memory with 1000s of images
BASE_DIRECTORY       = "/home/jorge/workspace/ncappzoo/"
PICTURES_DIRECTORY   = "/home/jorge/Pictures/"
graph_file           = BASE_DIRECTORY + "caffe/GoogLeNet/graph"
labels_file          = BASE_DIRECTORY + "data/ilsvrc12/synset_words.txt"
colormode            = "BGR"
FILE_EXTENSION       = "*.jpg"
mean                 = [104.00698793, 116.66876762, 122.67891434]
dim                  = (224, 224)
MAX_IMAGE_COUNT      =   25
scale                =    1

def pre_process_image():
    global image_folder

    img_list = []
    # Create a list of all files in current directory & sub-directories
    image_folder = filedialog.askdirectory(title="Choose a directory", \
                                           initialdir=os.path.dirname(PICTURES_DIRECTORY), mustexist=True)

    if (not image_folder):
        image_folder = PICTURES_DIRECTORY

    img_paths = [y for x in os.walk(image_folder)
                 for y in glob.glob(os.path.join(x[0], FILE_EXTENSION))]

    # Sort file names in alphabetical order
    img_paths.sort()

    for img_index, img_name in enumerate(img_paths):

        # Set a limit on the image count, so that it doesn't fill up the memory
        if img_index >= MAX_IMAGE_COUNT:
            break

        # Read & resize image [Image size is defined during training]
        img = skimage.io.imread(img_name)
        img = skimage.transform.resize(img, dim, preserve_range=True)

        # Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
        if(colormode == "BGR"):
            img = img[:, :, ::-1]

        # Mean subtraction & scaling
        img = img.astype(numpy.float32)
        img = (img - numpy.float32(mean)) * scale

        img_list.append(img)

    return img_list, img_paths

def infer_images(graph, img_list, img_paths):
    # Load the labels file
    labels =[line.rstrip('\n') for line in
                   open(labels_file) if line != 'classes\n']

    report = "Batch classifier results for: " + image_folder + "\n\nPredictions:\n"

    for img_index, img in enumerate(img_list):
        output, inference_time = mv.infer_image(graph, img)

        # Find the index of highest confidence
        top_prediction = output.argmax()

        # Print top predictions for each image
        report = report + ntpath.basename(img_paths[img_index]) \
                + " can be a " + labels[top_prediction][10:] \
                + " with " + str(int(100 * output[top_prediction])) \
                + "% confidence. (Time taken: " \
                + str(int((numpy.sum(inference_time)))) + "ms)\n"
    return report

def main():
    img_list, img_paths = pre_process_image()
    device = mv.open_ncs_device()
    graph  = mv.load_graph(device, graph_file)
    report = infer_images(graph, img_list, img_paths)
    mv.close_ncs_device(device, graph)
    return report

if __name__ == '__main__':
    print(main())


