#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory. 

from mvnc import mvncapi as mvnc
from tkinter.filedialog import askopenfilename
from PIL import Image
import sys
import numpy
import cv2
import time
import csv
import os
import sys

MAX_WIDTH     = 600

def assign_ncs_devices():
    '''
    use the first NCS device for tiny YOLO processing, and the rest for GoogLeNet processing
    '''
    global device

    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)
    devices = mvnc.enumerate_devices()

    if len(devices) < 1:
        print('This application requires an NCS device.')
        print('Insert one and try again!')
        return 1

    device = mvnc.Device(devices[0])
    device.open()

def execute_graph(graph_file, img):
	try:
		with open(graph_file, mode = 'rb') as f:
			graph_from_disk = f.read()
		graph = mvnc.Graph(graph_file)
		input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_from_disk)
	except:
		print('Error - could not load graph file')
		device.close()
		device.destroy()
		return 1

	graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, 'user object')
	output, userobj = output_fifo.read_elem()
	inference_time         = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)

	input_fifo.destroy()
	output_fifo.destroy()
	graph.destroy()
	device.close()
	device.destroy()

	return output, userobj, inference_time

def show_image(image_name):
    '''show_image handles displaying on the user screen at an appropriate size (since input images can vary wildly on
    their physical dimensions'''
    image = Image.open(image_name)
    (width, height) = image.size

    if width > MAX_WIDTH:
        scale_factor = MAX_WIDTH / width
        image = image.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)
    image.show()

    return image

def main():
	# categories for age and gender
	age_list    = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
	gender_list = ['Male','Female']

	# read in and pre-process the image:
	ilsvrc_mean = numpy.load("/home/jorge/workspace/ncappzoo/data/age_gender/age_gender_mean.npy").mean(1).mean(1)
	dim         = (227, 227)
	image_name  = askopenfilename()
	show_image(image_name)
	img = cv2.imread(image_name)

	img = cv2.resize(img, dim)
	img = img.astype(numpy.float32)
	img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
	img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
	img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

	# open the network graph file
	graph_file = "/home/jorge/workspace/ncappzoo/caffe/GenderNet/graph"
	assign_ncs_devices()

	#execute the network with the input image on the NCS
	output,userobj, inference_time = execute_graph(graph_file, img)
	order          = output.argsort()
	last           = len(order) - 1
	predicted      = int(order[last])

	report = "Gender Estimator results for:" + image_name + "\n\n"
	report = report + "Gender predicted: " + gender_list[predicted] + " with confidence of " + \
             str(int(100*output[predicted])) + " %\n"
	report = report + "\n  Evaluation time: " + str(int(numpy.sum(inference_time))) + " ms  "
	return report


