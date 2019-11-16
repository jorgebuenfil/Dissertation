#! /usr/bin/env python3
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

import cv2
import queue
import threading
import time

class video_processor:
    """
    Pulls images from video device and places them in a Queue if the queue is full will start to skip video frames.
    """
    def __init__(self, output_queue, video_file, queue_put_wait_max = 0.01,
                 request_video_width = 640, request_video_height = 480,
                 queue_full_sleep_seconds = 0.1):
        """
        Initializer for the class Video_Processor.
        :param output_queue: instance of queue.Queue in which the video frame images will be placed.
        :param video_file: video file to read.
        :param queue_put_wait_max: max number of seconds to wait when putting images into the output queue.
        :param request_video_width: default value  = 640.
        :param request_video_height: default value = 480.
        :param queue_full_sleep_seconds: number of seconds to sleep when the output queue is full.
        """
        self._queue_full_sleep_seconds = queue_full_sleep_seconds
        self._queue_put_wait_max       = queue_put_wait_max
        self._video_file               = video_file
        self._request_video_width      = request_video_width
        self._request_video_height     = request_video_height
        self._pause_mode               = False

        # create the video device
        self._video_device = cv2.VideoCapture(self._video_file)

        if ((self._video_device == None) or (not self._video_device.isOpened())):
            print('\n\n')
            print('Error - could not open video device.')
            print('If you installed python opencv via pip or pip3 you')
            print('need to uninstall it and install from source with -D WITH_V4L=ON')
            print('Use the provided script: install-opencv-from_source.sh')
            print('\n\n')
            return

        # Request the dimensions
        self._video_device.set(cv2.CAP_PROP_FRAME_WIDTH, self._request_video_width)
        self._video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, self._request_video_height)

        # save the actual dimensions
        self._actual_video_width  = self._video_device.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._actual_video_height = self._video_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('actual video resolution: ' + str(self._actual_video_width) + ' x ' + str(self._actual_video_height))

        self._output_queue  = output_queue
        self._worker_thread = threading.Thread(target = self._do_work, args = ())

    def get_actual_video_width(self):
        "Width of the images that will be put in the queue"
        return self._actual_video_width

    def get_actual_video_height(self):
        "Height of the images that will be put in the queue"
        return self._actual_video_height

    def start_processing(self):
        "start reading from the video and placing images in the output queue"
        self._end_flag = False
        self._worker_thread.start()

    def stop_processing(self):
        "stop reading from video device and placing images in the output queue"
        self._end_flag = True
        self._worker_thread.join()

    def pause(self):
        self._pause_mode = True

    def unpause(self):
        self._pause_mode = False

    def _do_work(self):
        """
        Thread target.  when call start_processing this function will be called in its own thread.
        Tt will keep working until stop_processing is called or an error is encountered.
        :return: none.
        """
        if (self._video_device == None):
            print('video_processor _video_device is None, returning.')
            return

        while (not self._end_flag):
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                ret_val, input_image = self._video_device.read()

                if (not ret_val):
                    print("No image from video device, exiting")
                    break
                self._output_queue.put(input_image, True, self._queue_put_wait_max)
            except queue.Full:
                """
                The video device is probably way faster than the processing so if our output queue 
                is full sleep a little while before trying the next image from the video.
                """
                time.sleep(self._queue_full_sleep_seconds)

    def cleanup(self):
        """
        Should be called once for each class instance when finished with it.
        :return: _video_device.release
        """
        self._video_device.release()
