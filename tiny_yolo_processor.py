#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# processes images via tiny yolo

from mvnc import mvncapi as mvnc
import numpy as np
import threading
import queue
import cv2

class tiny_yolo_processor:
    # Tiny Yolo assumes input images are these dimensions.
    TY_NETWORK_IMAGE_WIDTH  = 448
    TY_NETWORK_IMAGE_HEIGHT = 448

    def __init__(self, tiny_yolo_graph_file, ncs_device, input_queue, output_queue,
                 inital_box_prob_thresh, initial_max_iou, queue_wait_input, queue_wait_output):
        '''
    Initializes an instance of the class.
        :param tiny_yolo_graph_file:  path and filename to the tiny YOLO graph file created by the NCSDK compiler.
        :param ncs_device: open NCS device object.
        :param input_queue: queue object from which images will be pulled and inferences processed on.
        :param output_queue: queue object on which the tiny YOLO inference results will be placed.
          Each result will add to the queue the opencv image on which the inference was run, and a list with:
                - string that is network classification ie 'cat', or 'chair' etc
                - float value for box center X pixel location within source image
                - float value for box center Y pixel location within source image
                - float value for box width in pixels within source image
                - float value for box height in pixels within source image
                - float value that is the probability for the network classification.
        :param inital_box_prob_thresh: initial box probability threshold for boxes returned from the inferences,
        :param initial_max_iou: inital value for the max iou which determines duplicate boxes using Itersection over Union.
        :param queue_wait_input:
        :param queue_wait_output:
        '''
        self._queue_wait_input  = queue_wait_input
        self._queue_wait_output = queue_wait_output

        # Load tiny_yolo graph from disk and allocate graph via API
        try:
            with open(tiny_yolo_graph_file, mode='rb') as ty_file:
                ty_graph_from_disk = ty_file.read()
            self._ty_graph = mvnc.Graph(tiny_yolo_graph_file)
            self.input_fifo, self.output_fifo = self._ty_graph.allocate_with_fifos(ncs_device, ty_graph_from_disk)

        except:
            print('\n\n')
            print('Error - could not load tiny yolo graph file: ' + tiny_yolo_graph_file)
            print('\n\n')
            raise

        self._box_probability_threshold = inital_box_prob_thresh
        self._max_iou                   = initial_max_iou
        self._input_queue               = input_queue
        self._output_queue              = output_queue
        self._worker_thread             = threading.Thread(target=self._do_work, args=())

    def cleanup(self):
        """
        Called when done with the instance of the class
        :return: none
        """
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        self._ty_graph.destroy()

    def start_processing(self):
        """
        Start asynchronous processing of the images on the input queue via
        a worker thread and place inference results on the output queue.
        :return: none
        """
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())

        self._worker_thread.start()

    def stop_processing(self):
        """"
        Stop asynchronous processing of the images on input queue when returns the worker thread will be terminated.
        """
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    def do_inference(self, input_image):
        """
        Single inference.
        :param input_image: input_image is the image on which to run the inference. It can be any size.
        :return:
                - result from _filter_objects() which is a list of lists. Each of the inner lists represent
                 one found object and contain:
                     - string that is network classification ie 'cat', or 'chair' etc.
                     - float value for box center X pixel location within input_image.
                     - float value for box center Y pixel location within input_image.
                     - float value for box width in pixels within input_image.
                     - float value for box height in pixels within input_image.
                     - float value that is the probability for the network classification.
        """
        # save original width and height
        input_image_width  = input_image.shape[1]
        input_image_height = input_image.shape[0]

        """
        Resize image to network width and height, then convert to float32, normalize 
        (divide by 255), and finally convert to float32 to pass to LoadTensor as input
        for an inference this returns a new image so the input_image is unchanged
        """
        inference_image = cv2.resize(input_image,
                                 (tiny_yolo_processor.TY_NETWORK_IMAGE_WIDTH,
                                  tiny_yolo_processor.TY_NETWORK_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        # modify inference_image for TinyYolo input
        inference_image = inference_image[:, :, ::-1]  # convert to RGB
        inference_image = inference_image.astype(np.float32)
        inference_image = np.divide(inference_image, 255.0)

        # Load tensor and get result.  This executes the inference on the NCS
        self._ty_graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, inference_image, None)
        output, userobj = self.output_fifo.read_elem()

        # filter out all the objects/boxes that don't meet thresholds
        return self._filter_objects(output.astype(np.float32), input_image_width, input_image_height)

    def _do_work(self):
        """
        Worker thread which handles the asynchronous processing of images on the input queue,
        running inferences on the NCS and placing results on the output queue
        :return: none.
        """
        while (not self._end_flag):
            try:
                # get input image from input queue.  This does not copy the image
                input_image   = self._input_queue.get(True, self._queue_wait_input)

                # get the inference and filter etc.
                filtered_objs = self.do_inference(input_image)

                # put the results along with the input image on the output queue
                self._output_queue.put((input_image, filtered_objs), True, self._queue_wait_output)

                # finished with this input queue work item
                self._input_queue.task_done()

            except queue.Empty:
                print('ty_proc, input queue empty')
            except queue.Full:
                print('ty_proc, output queue full')

    def get_box_probability_threshold(self):
        """ Get the box probability threshold. It will be between 0.0 and 1.0
        Higher numbers will result in less boxes returned during inferences
        :return: _box_probability_threshold
        """
        return self._box_probability_threshold

    def set_box_probability_threshold(self, value):
        """
        Set the box probability threshold.
        :param value: new value, it must be between 0.0 and 1.0
        lower values will allow less certain boxes in the inferences which will result
        in more boxes per image.  Higher values will filter out less certain boxes
        and result in fewer boxes per inference.
        :return: _box_probability_threshold
        """
        self._box_probability_threshold = value

    def get_max_iou(self):
        """
        return the current max intersection-over-union threshold value to use
        when determining duplicate boxes in an inference.
        objects/boxes found that produce IoU values over this threshold will be
        considered the same object when filtering the Tiny Yolo inference output.
        :return: max_iou
        """
        return self._max_iou

    def set_max_iou(self, value):
        """
        Sets a new max intersection-over-union threshold value to use when determining
        duplicate boxes in an inference.
        Objects/boxes found that produce iou values over this threshold will be considered
        the same object when filtering the Tiny Yolo inference output.
        :param value:
        :return: _max_iou
        """
        self._max_iou = value

    def _filter_objects(self, inference_result, input_image_width, input_image_height):
        """
        Interpret the output from a single inference of TinyYolo (GetResult) and filter out
        objects/boxes with low probabilities.
        :param inference_result: array of floats returned from the API GetResult but converted to float32 format.
        :param input_image_width: width of the input image.
        :param input_image_height: height of the input image.
        :return: list of lists. each of the inner lists represent one found object and contain:
        - string that is network classification ie 'cat', or 'chair' etc.
        - float value for box center X pixel location within source image.
        - float value for box center Y pixel location within source image.
        - float value for box width in pixels within source image.
        - float value for box height in pixels within source image.
        - float value that is the probability for the network classification.
        """
        # the raw number of floats returned from the inference (GetResult())
        num_inference_results = len(inference_result)

        # the 20 classes this network was trained on
        network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                   "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                   "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

        # which types of objects do we want to include.
        network_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1]

        num_classifications = len(network_classifications) # should be 20
        grid_size           = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
        boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell
        """
        grid_size is 7 (grid is 7x7)
        num classifications is 20
        boxes per grid cell is 2
        """
        all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

        """
        classification_probabilities  contains a probability for each classification for each 64x64 
        pixel square of the grid.  The source image contains 7x7 of these 64x64 pixel squares.
        There are 20 possible classifications.
        """
        classification_probabilities = \
            np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
        num_of_class_probs = len(classification_probabilities)

        # The probability scale factor for each box
        box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

        # get the boxes from the results and adjust to be pixel units
        all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
        self._boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

        # adjust the probabilities with the scaling factor
        for box_index in range(boxes_per_grid_cell): # loop over boxes
            for class_index in range(num_classifications): # loop over classifications
                all_probabilities[:,:,box_index,class_index] = \
                    np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])

        probability_threshold_mask      = np.array(all_probabilities >= self._box_probability_threshold, dtype='bool')
        box_threshold_mask              = np.nonzero(probability_threshold_mask)
        boxes_above_threshold           = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        probabilities_above_threshold   = all_probabilities[probability_threshold_mask]

        # sort the boxes from highest probability to lowest and then sort the probabilities and classifications to match
        argsort                         = np.array(np.argsort(probabilities_above_threshold))[::-1]
        boxes_above_threshold           = boxes_above_threshold[argsort]
        classifications_for_boxes_above = classifications_for_boxes_above[argsort]
        probabilities_above_threshold   = probabilities_above_threshold[argsort]

        # get mask for boxes that seem to be the same object
        duplicate_box_mask = self._get_duplicate_box_mask(boxes_above_threshold)

        # update the boxes, probabilities and classifications removing duplicates.
        boxes_above_threshold           = boxes_above_threshold[duplicate_box_mask]
        classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
        probabilities_above_threshold   = probabilities_above_threshold[duplicate_box_mask]

        classes_boxes_and_probs = []
        for i in range(len(boxes_above_threshold)):
            if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
                classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

        return classes_boxes_and_probs

    def _get_duplicate_box_mask(self, box_list):
        """
        Creates a mask to remove duplicate objects (boxes) and their related probabilities
        and classifications that should be considered the same object.  This is determined by
        how similar the boxes are based on the intersection-over-union metric.
        :param box_list: list of boxes (4 floats for centerX, centerY and Length and Width)
        :return: filter_iou_mask
        """
        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self._get_intersection_over_union(box_list[i], box_list[j]) > self._max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask

    @staticmethod
    def _boxes_to_pixel_units(box_list, image_width, image_height, grid_size):
        """
        Converts the boxes in box list to pixel units assumes box_list is the output from
        the box output from the tiny yolo network and is [grid_size x grid_size x 2 x 4].
        :param box_list:
        :param image_width:
        :param image_height:
        :param grid_size:
        :return: none
        """
        # number of boxes per grid cell
        boxes_per_cell = 2

        # setup some offset values to map boxes to pixels
        # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
        box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

        # adjust the box center
        box_list[:,:,:,0] += box_offset
        box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
        box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

        # adjust the lengths and widths
        box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
        box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

        #scale the boxes to the image size in pixels
        box_list[:,:,:,0] *= image_width
        box_list[:,:,:,1] *= image_height
        box_list[:,:,:,2] *= image_width
        box_list[:,:,:,3] *= image_height

    @staticmethod
    def _get_intersection_over_union(box_1, box_2):
        """
        Evaluates the intersection-over-union for two boxes.
        The intersection-over-union metric determines how close two boxes are to being the same box.
        The closer the boxes are to being the same, the closer the metric will be to 1.0
        :param box_1: array of 4 numbers which are the (x, y) points that define the center of the box
        and the length and width of the box.
        :param box_2: array of 4 numbers which are the (x, y) points that define the center of the box
        and the length and width of the box.
        :return: intersection-over-union (between 0.0 and 1.0) for the two boxes specified.
        """
        # one diminsion of the intersecting box
        intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                             max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

        # the other dimension of the intersecting box
        intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                             max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
            # no intersection area
            intersection_area = 0
        else :
            # intersection area is product of intersection dimensions
            intersection_area =  intersection_dim_1*intersection_dim_2

        # calculate the union area which is the area of each box added
        # and then we need to subtract out the intersection area since
        # it is counted twice (by definition it is in each box)
        union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area

        # now we can return the intersection over union
        iou = intersection_area / union_area

        return iou
