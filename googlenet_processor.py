#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# processes images via googlenet

from mvnc import mvncapi as mvnc
import numpy as np
import threading
import queue
import cv2

class googlenet_processor:
    '''
    Manager of GoogLeNet processing.
    '''
    # GoogLeNet assumes input images are these dimensions
    GN_NETWORK_IMAGE_WIDTH  = 224
    GN_NETWORK_IMAGE_HEIGHT = 224

    MEAN_FILE_NAME   = "/home/jorge/workspace/ncappzoo/data/ilsvrc12/ilsvrc_2012_mean.npy"
    LABELS_FILE_NAME = "/home/jorge/workspace/ncappzoo/data/ilsvrc12/synset_words.txt"

    def __init__(self, googlenet_graph_file, ncs_device, input_queue, output_queue,
                 queue_wait_input, queue_wait_output):
        '''
    Initialize the class instance

        :param googlenet_graph_file: path and filename of the GoogLeNet graph file produced via the NCSDK compiler.
        :param ncs_device: open Device instance from the NCSDK.
        :param input_queue: queue instance from which images will be pulled that are in turn processed
        (inferences are run on) via the NCS device. Each item on the queue should be an opencv image.
        it will be resized as needed for the network
        :param output_queue: queue object on which the results of the inferences will be placed. For each inference
        a list of the following items will be placed on the output_queue:

             - index of the most likely classification from the inference.
             - label for the most likely classification from the inference.
             - probability the most likely classification from the inference.
        :param queue_wait_input:
        :param queue_wait_output:
        '''
        self._queue_wait_input  = queue_wait_input
        self._queue_wait_output = queue_wait_output

        # GoogLeNet mean values will be read in from .npy file
        self._gn_mean = [0., 0., 0.]

        # labels to display along with boxes if googlenet classification is good
        # these will be read in from the synset_words.txt file for ilsvrc12
        self._gn_labels = [""]

        # loading the means from file
        try:
            self._gn_mean = np.load(googlenet_processor.MEAN_FILE_NAME).mean(1).mean(1)
        except:
            print('\n\nError - could not load means from ' + googlenet_processor.MEAN_FILE_NAME)
            print('\n\n')
            raise

        # loading the labels from file
        try:
            self._gn_labels = np.loadtxt(googlenet_processor.LABELS_FILE_NAME, str, delimiter='\t')
            for label_index in range(0, len(self._gn_labels)):
                temp = self._gn_labels[label_index].split(',')[0].split(' ', 1)[1]
                self._gn_labels[label_index] = temp
        except:
            print('\n\nError - could not read labels from: ' + googlenet_processor.LABELS_FILE_NAME)
            print('\n\n')
            raise

        # Load googlenet graph from disk and allocate graph via API
        try:
            with open(googlenet_graph_file, mode='rb') as gn_file:
                gn_graph_from_disk = gn_file.read()
            self._gn_graph = mvnc.Graph(googlenet_graph_file)
            self.input_fifo, self.output_fifo = self._gn_graph.allocate_with_fifos(ncs_device, gn_graph_from_disk)
        except:
            print('\n\nError - could not load GoogLeNet graph file: ' + googlenet_graph_file)
            print('\n\n')
            raise

        self._input_queue   = input_queue
        self._output_queue  = output_queue
        self._worker_thread = threading.Thread(target = self._do_work, args = ())

    def cleanup(self):
        "called once the instance no longer used."
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        self._gn_graph.destroy()

    def start_processing(self):
        """
        Starts asynchronous processing on a worker thread that will pull images
        off the input queue and placing results on the output queue.
        :return: none.
        """
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())
        self._worker_thread.start()

    def stop_processing(self):
        """
        Stops asynchronous processing of the worker thread upon return the worker thread will have terminated.
        :return: none.
        """
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    def _do_work(self):
        """
        Worker Thread function. called when start_processing is called and returns when stop_processing is called.
        :return: none.
        """
        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, self._queue_wait_input)
                index, label, probability = self.googlenet_inference(input_image, "NPS")
                self._output_queue.put((index, label, probability), True, self._queue_wait_output)
                self._input_queue.task_done()
            except queue.Empty:
                print('GoogLeNet processor: No more images in queue.')
            except queue.Full:
                print('GoogLeNet processor: queue full')

    def googlenet_inference(self, input_image):
        """
        Executes an inference using the googlenet graph and image passed.
        :param input_image: image on which a googlenet inference should be executed.
        It will be resized to match googlenet image size requirements and also converted to float32.
        Returns a list of the following three items:
            - index of the most likely classification from the inference.
            - label for the most likely classification from the inference.
            - probability the most likely classification from the inference.
        :param user_obj: Not used.
        :return: none
        """
        # Resize image to googlenet network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
        input_image = cv2.resize(input_image, (googlenet_processor.GN_NETWORK_IMAGE_WIDTH,
                                               googlenet_processor.GN_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        input_image = input_image.astype(np.float32)
        input_image[:, :, 0] = (input_image[:, :, 0] - self._gn_mean[0])
        input_image[:, :, 1] = (input_image[:, :, 1] - self._gn_mean[1])
        input_image[:, :, 2] = (input_image[:, :, 2] - self._gn_mean[2])

        # Load tensor and get result.  This executes the inference on the NCS
        _gn_graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, input_image, None)
        output, userobj = self.output_fifo.read_elem()

        order = output.argsort()[::-1][:1]

        print('\n------- Prediction --------')
        for i in range(0, 5):
            print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self._gn_labels[
                order[i]] + '  label index is: ' + str(order[i]))

        # index, label, probability
        return order[0], self._gn_labels[order[0]], output[order[0]]
