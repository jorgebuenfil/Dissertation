#!/usr/bin/python3
"""
Using Movidius Neural Compute API (mvncapi version 2)
    by Jorge Buenfil
"""
import mvnc.mvncapi as mvnc
import tkinter as tk
from tkinter import messagebox

def open_ncs_device(number_of_devices=1):
    # Configure the NCS verbosity reports
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)
    # Look for enumerated NCS device(s); quit program if none found.
    device_list = mvnc.enumerate_devices()
    if len(device_list) < number_of_devices:
        messagebox.showerror("NCS Error", "Not enough devices available")
        return
    device1 = mvnc.Device(device_list[0])
    device1.open()
    if number_of_devices   == 1:
        return device1
    elif number_of_devices == 2:
        device2 = mvnc.Device(device_list[1])
        device2.open()
        return device1, device2

def load_graph(device, graph_file):
    global input_fifo, output_fifo

    with open(graph_file, mode='rb') as f:
        graph_buffer          = f.read()
    graph                     = mvnc.Graph(graph_file)
    input_fifo, output_fifo   = graph.allocate_with_fifos(device, graph_buffer)

    return graph,

def load_multidevice_graph(device, graph_file):
    """
    Function to work with 2 NCS devices.

    :param device: NCS device to use
    :param graph_file: graph file for the particular device
    :return: opened graph object, input_fifo, and output_fifo for the device
    """
    with open(graph_file, mode='rb') as f:
        graph_buffer          = f.read()
    graph                     = mvnc.Graph(graph_file)
    external_input_fifo, external_output_fifo   = graph.allocate_with_fifos(device, graph_buffer)

    return graph, external_input_fifo, external_output_fifo

def infer_image(graph, img):
    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
    output, userobj = output_fifo.read_elem()
    inference_time  = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
    return output, inference_time

def infer_multidevice_image(graph, img, external_input_fifo, external_output_fifo):
    graph.queue_inference_with_fifo_elem(external_input_fifo, external_output_fifo, img, None)
    output, userobj = external_output_fifo.read_elem()
    inference_time  = graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
    return output, inference_time, external_input_fifo, external_output_fifo

def close_ncs_device(device, graph):
    input_fifo.destroy()
    output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()

def close_multidevice_ncs_device(device, graph, external_input_fifo, external_output_fifo):
    external_input_fifo.destroy()
    external_output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()

def close_only_ncs_device(device):
    """
    close only ncs_device (no graph)
    :param device: NCS device
    :return: none.
    """
    device.close()
    device.destroy()

def test_all_ncs_devices():
    "Test all available ncs_devices"
    msg               = ""
    device_list       = mvnc.enumerate_devices()
    number_of_devices = len(device_list)
    opened_devices = 0
    closed_devices = 0

    if number_of_devices < 1:
        msg ="No devices available found\n"
        return msg

    for i in range(number_of_devices):
        try:
            device = mvnc.Device(device_list[i])
            device.open()
            opened_devices += 1
            msg = msg + "Device # " + str(i+1) + " opened fine\n"
        except:
            msg = msg + "Error - Could not open NCS device # " + str(i+1) + "\n"

        try:
            device.close()
            device.destroy()
            closed_devices += 1
            msg = msg + "Device # " + str(i+1) + " closed fine\n-------------------\n"
        except:
            msg = msg + "Error - could not close NCS device # " + str(i+1) + "\n"

    msg = msg + "\n\n" + str(opened_devices) + " device(s)  opened OK!"
    msg = msg + "\n"   + str(closed_devices) + " device(s)  closed OK!"

    return msg

def manage_NCS(graph_file, image):
        '''
        Operate the hardware co-processor Neural Compute Stick (NCS) by Intel Movidius to accept a graphical
        program of an already trained convolutional neural network and run examine an image to produce
        classification results.
        '''
        device = open_ncs_device()
        graph  = load_graph(device, graph_file)
        output, inference_time = infer_image(graph, image)
        close_ncs_device(device, graph)
        return output, inference_time

