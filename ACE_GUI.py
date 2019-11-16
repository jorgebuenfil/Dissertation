# !/usr/bin/python3
# GUI_tests.py

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import tensorflow as tf
from glob import glob
import face_recognition
from PIL import Image
import numpy as np
import subprocess
import pickle
import imutils
from imutils.video import VideoStream
import time
import csv
import cv2
import os
import skimage.io
import skimage.transform
from utils import visualize_output
from utils import deserialize_output

# imported libraries created or significantly modified by Jorge
import vgg16
import security_cam
import gender_estimator
import video_object_detector
import build_face_dataset
import stream_ty_gn                            as styg
import mvnc_functions                          as mv
import recognize_faces_video                   as videofaces
import Batch_Classifier.rapid_image_classifier as rapid_image_classifier
import live_image_classifier
import log_image_classifier

# Global Definitions
MAX_WIDTH     = 500
main_window   = tk.Tk()
filepaths     = []
default_directory      = "/home/jorge/Pictures/"
default_image          = default_directory + "rainha.jpg"
default_file_extension = "*.jpg"
IMAGE_EXTENSION        = "*.jpg"
IMAGE_EXTENSION2       = "*.jpeg"


PI_BASE_DIRECTORY      = "/home/jorge/share/"
pi_image_name = "/Pictures/capture.jpg"
Alpha_image   = PI_BASE_DIRECTORY + "Alpha"   + pi_image_name
Beta_image    = PI_BASE_DIRECTORY + "Beta"    + pi_image_name
Charlie_image = PI_BASE_DIRECTORY + "Charlie" + pi_image_name
Delta_image   = PI_BASE_DIRECTORY + "Delta"   + pi_image_name

class Input_Image:
    """Common base class for all images for classification and detection purposes"""

    def __init__(self):
        'Class constructor to set the path to the image and its name'
        self.name = name

    def image_resize(image,    kind="PIL", title=""):
        '''
        show_image handles displaying on the user screen at an appropriate size
        (since input images can vary wildly on their physical dimensions
        '''
        if kind  == "PIL":
            (width, height) = image.size
        elif kind == "CV2":
            (width, height) = image.shape[:2]
        else:
            error_msg = "Image type not recognized"
            messagebox.showerror("Error", error_msg)

        if width > MAX_WIDTH:
            scale_factor = MAX_WIDTH / width
        else:
            scale_factor = 1.0

        if kind   == "PIL":
            image = image.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)
            image.show()
        elif kind =="CV2":
            image = cv2.resize(image, (int(height * scale_factor), int(width * scale_factor)))
            cv2.imshow(title, image)
            cv2.waitKey(50)

        return image

    def show_image(image_name, kind="PIL", title=""):
        '''
        Open and show_image handles with PILLOW or CV2 displaying on the user screen at an appropriate size
        (since input images can vary wildly on their physical dimensions
        '''
        if kind   == "PIL":
            image = Image.open(image_name)
            image = Input_Image.image_resize(image)
        elif kind=="CV2":
            image = cv2.imread(image_name)
            image = Input_Image.image_resize(image, kind, title)

        return image

    def preprocess(image_name, dimensions):
        image = cv2.imread(image_name)
        image = cv2.resize(image, dimensions)
        image = image.astype(np.float32)
        return image

    def load_labels(input_file):
        labels = []
        with open(input_file, 'r') as f:
            for line in f:
                cat = line.split('\n')[0]
                if cat != 'classes':
                    labels.append(cat)
            f.close()
        return labels

    def yolo():
        "You Only Look Once (YOLO): Example of an object detector using a Convolutional Neural Network"
        os.chdir("/home/jorge/Documents/Dissertation/Demo/scripts/")
        messagebox.showinfo("YOLO processing", "Click button to start, then wait  a few seconds...")
        image_name   = App.ask_filename()
        subprocess.run(["./2-Yolo.sh", image_name])
        results_file = open("/home/jorge/workspace/darknet/yolo_results.txt", "r")
        results      = results_file.read()
        results_file.close()
        App.show_message("Results", "YOLO results\n\n" + results)

    def handle_tensorflow_single():
        Input_Image.tensorflow1(False, "MobileNet")

    def handle_tensorflow_batch():
        Input_Image.tensorflow1(True, "MobileNet")

    def tensorflow1(batch_operation=False, model="inception_v3"):
        """
        Image classification with TensorFlow using a trained Convolutional Neural Network that identifies pistols,
        rifles, bullets, knives, and persons.
        This application uses Knowledge Transfer from GoogLeNet (Inception v3) or MobileNet.
        MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases.
        Not a detector so no boxes are placed on top of detected object.
        """
        BASE_DIRECTORY       = "/home/jorge/workspace/TensorFlow_Projects/tensorflow-for-poets-2/tf_files/"
        if model == "MobileNet":
            IMAGE_SIZE  = 224
            GRAPH_INPUT_OPERATION = "import/input"
            model_file  = BASE_DIRECTORY + "MobileNet_1.0_224_graph.pb"
            labels_file = BASE_DIRECTORY + "MobileNet_1.0_224_labels.txt"
        else:
            IMAGE_SIZE  = 299
            GRAPH_INPUT_OPERATION = "import/Mul"
            model_file  = BASE_DIRECTORY + "inception_v3_graph.pb"
            labels_file = BASE_DIRECTORY + "inception_v3_labels.txt"

        CONFIDENCE_LEVEL = 0.5
        result_number    = 1

        def load_labels(label_file):
            label = []
            proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
            for l in proto_as_ascii_lines:
                label.append(l.rstrip())
            return label

        def load_graph(model_file):
            graph     = tf.Graph()
            graph_def = tf.GraphDef()

            with open(model_file, "rb") as f:
                graph_def.ParseFromString(f.read())
            with graph.as_default():
                tf.import_graph_def(graph_def)

            return graph

        graph  = load_graph(model_file)
        labels = load_labels(labels_file)

        def read_tensor_from_image_file(file_name, input_height, input_width,
                                                   input_mean=128, input_std=128):
            """
            Inception v3 uses images of height, width              = 299.
            MobileNet and Inception v1 use images of height, width = 224.

            :param file_name:
            :param input_height:
            :param input_width:
            :param input_mean:
            :param input_std:
            :return: result of classification done on the provided image.
            """
            input_name       = "file_reader"
            output_name      = "normalized"
            file_reader      = tf.read_file(file_name, input_name)

            if file_name.endswith(".png"):
                image_reader = tf.image.decode_png(file_reader,  channels=3,  name='png_reader')
            elif file_name.endswith(".gif"):
                image_reader = tf.squeeze(tf.image.decode_gif(file_reader,   name='gif_reader'))
            elif file_name.endswith(".bmp"):
                image_reader = tf.image.decode_bmp(file_reader,              name='bmp_reader')
            else:
                image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

            float_caster     = tf.cast(image_reader, tf.float32)
            dims_expander    = tf.expand_dims(float_caster, 0)
            resized          = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized       = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
            sess             = tf.Session()
            result           = sess.run(normalized)

            return result

        if not batch_operation:
            image_filename = App.ask_filename()
            Input_Image.show_image(image_filename)

            start   = time.time()

            # Process an image
            t = read_tensor_from_image_file(image_filename, input_height=IMAGE_SIZE, input_width=IMAGE_SIZE)
            input_operation  = graph.get_operation_by_name(GRAPH_INPUT_OPERATION)
            output_operation = graph.get_operation_by_name("import/final_result")

            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
            results = np.squeeze(results)
            top_k   = results.argsort()[-5:][::-1]

            end     = time.time()
            report ="Tensor Flow (pre-trained) results for image:\n" + image_filename +"\n\n"

            for i in top_k:
                if results[i] > CONFIDENCE_LEVEL:
                    report = report + "  " + str(result_number) + ") " + str(round(results[i] * 100, 1)) + \
                             "% " + labels[i] + "  \n"
                result_number += 1

            report = report + "\n  Evaluation time: " + str(int(1000 * (end - start))) + " ms  "
            App.show_message("Tensor Flow (pre-trained) results", report)
        else:
            image_list = App.compile_image_list()
            for index, image_filename in enumerate(image_list):
                t = read_tensor_from_image_file(image_filename, input_height=IMAGE_SIZE, input_width=IMAGE_SIZE)

                input_operation  = graph.get_operation_by_name(GRAPH_INPUT_OPERATION)
                output_operation = graph.get_operation_by_name("import/final_result")

                with tf.Session(graph=graph) as sess:
                    results = sess.run(output_operation.outputs[0],
                                       {input_operation.outputs[0]: t})
                results = np.squeeze(results)
                top_k   = results.argsort()[-5:][::-1]

                with open('jorges_TF_inferences.csv', 'a', newline='\n') as csvfile:
                    inference_log = csv.writer(csvfile,  delimiter='@',
                                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for i in top_k:
                        if results[i] > CONFIDENCE_LEVEL:
                            inference_log.writerow([image_list[index], (results[i] * 100), labels[i]])
                        result_number += 1
            App.show_message("Results", "Inference complete! View results in './jorges_TF_inferences.csv'.")

    def alexnet():
        BASE_DIRECTORY = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/"
        graph_file  = BASE_DIRECTORY + "caffe/AlexNet/graph"
        mean_file   = BASE_DIRECTORY + "data/ilsvrc12/ilsvrc_2012_mean.npy"
        labels_file = BASE_DIRECTORY + "data/ilsvrc12/synset_words.txt"
        labels      = np.loadtxt(labels_file, str, delimiter='\t')

        # Show and resize the image prior to processing
        image_name  = App.ask_filename()
        Input_Image.show_image(image_name)
        image       = Input_Image.preprocess(image_name, (227, 227))
        ilsvrc_mean = np.load(mean_file).mean(1).mean(1)
        image[:, :, 0] = (image[:, :, 0] - ilsvrc_mean[0])
        image[:, :, 1] = (image[:, :, 1] - ilsvrc_mean[1])
        image[:, :, 2] = (image[:, :, 2] - ilsvrc_mean[2])

        # Run neural network using the hardware co-processor NCS
        output, inference_time = mv.manage_NCS(graph_file, image)
        # Report results to user
        order  = output.argsort()[::-1][:6]
        report ="AlexNet results for:" + image_name + "\n\n"
        for i in range(0, 3):
            report = report + "(" + str(i+1) + ") " + str(round(output[order[i]]*100, 1)) + "% " + "is some sort of " + \
                     labels[order[i]][10:len(labels[order[i]])] + "  \n"
        report = report + "\n  Evaluation time: " + str(int(np.sum(inference_time))) + " ms  "
        App.show_message("Results",report)

    def googlenet():
        BASE_DIRECTORY = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/"
        graph_file  = BASE_DIRECTORY + "caffe/GoogLeNet/graph"
        mean_file   = BASE_DIRECTORY + "data/ilsvrc12/ilsvrc_2012_mean.npy"
        labels_file = BASE_DIRECTORY + "data/ilsvrc12/synset_words.txt"
        labels      = np.loadtxt(labels_file, str, delimiter='\t')

        # Show and resize the image prior to processing
        image_name  = App.ask_filename()
        Input_Image.show_image(image_name)
        image       = Input_Image.preprocess(image_name, (224, 224))

        ilsvrc_mean = np.load(mean_file).mean(1).mean(1)
        image[:, :, 0] = (image[:, :, 0] - ilsvrc_mean[0])
        image[:, :, 1] = (image[:, :, 1] - ilsvrc_mean[1])
        image[:, :, 2] = (image[:, :, 2] - ilsvrc_mean[2])

        # Run neural network using the hardware co-processor NCS
        output, inference_time = mv.manage_NCS(graph_file, image)
        order = output.argsort()[::-1][:6]

        # Report results to user
        report = "GoogLeNet results for: " + image_name + "\n\n"
        for i in range(0, 3):
            report = report + "(" + str(i + 1) + ") " + \
                     str(round(output[order[i]] * 100, 1)) + \
                     "% " + "is some sort of " + \
                     labels[order[i]][10:len(labels[order[i]])] + "  \n"
        report = report + "\n  Evaluation time: " + str(int(np.sum(inference_time))) + " ms  "

        App.show_message("GoogLeNet Results",report)

    def squeezenet(get_image=True, display_results=True):
        BASE_DIRECTORY = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/"
        graph_file  = BASE_DIRECTORY + "caffe/SqueezeNet/graph"
        mean_file   = BASE_DIRECTORY + "data/ilsvrc12/ilsvrc_2012_mean.npy"
        labels_file = BASE_DIRECTORY + "data/ilsvrc12/synset_words.txt"
        labels      = np.loadtxt(labels_file, str, delimiter='\t')

        # Show and resize the image prior to processing
        image_name  = App.ask_filename()
        Input_Image.show_image(image_name)
        image       = Input_Image.preprocess(image_name, (227, 227))
        ilsvrc_mean = np.load(mean_file).mean(1).mean(1)
        image[:, :, 0] = (image[:, :, 0] - ilsvrc_mean[0])
        image[:, :, 1] = (image[:, :, 1] - ilsvrc_mean[1])
        image[:, :, 2] = (image[:, :, 2] - ilsvrc_mean[2])

        # Run neural network using the hardware co-processor NCS
        output, inference_time = mv.manage_NCS(graph_file, image)
        order = output.argsort()[::-1][:6]
        # Report results to user
        report = "SqueezeNet results for:" + image_name + "\n\n"
        for i in range(0, 3):
            report = report + "(" + str(i + 1) + ") " + str(round(output[order[i]] * 100, 1)) + "% " + "is some sort of " + \
                     labels[order[i]][10:len(labels[order[i]])] + "  \n"
        report = report + "\n  Evaluation time: " + str(int(np.sum(inference_time))) + " ms  "
        App.show_message("Results", report)

    def triple_classification():
        os.chdir("/home/jorge/workspace/ncappzoo/apps/classifier-gui/")
        subprocess.run(["./classifier_gui.py"])

    def inception_v1():
        "Inception v1 example using the hardware co-processor Neural Compute Stick (NCS) by Intel Movidius"
        graph_file  = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/tensorflow/inception_v1/graph"
        labels_file = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/tensorflow/inception_v1/categories.txt"
        mean        = 128
        std         = 1/128

        # Show and resize the image prior to processing
        image_name  = App.ask_filename()
        Input_Image.show_image(image_name)
        image       = Input_Image.preprocess(image_name, (224, 224))
        # Image normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean) * std

        labels = Input_Image.load_labels(labels_file)
        # Run neural network using the hardware co-processor NCS
        output, inference_time = mv.manage_NCS(graph_file, image)
        order = output.argsort()[::-1][:6]
        # Report results to user
        report = "Inception v1 results for:" + image_name + "\n\n"
        for i in range(3):
            report = report + "(" + str(i + 1) + ") " + str(round(output[order[i]] * 100, 1)) + "% " + "is some sort of " + \
                     labels[order[i]] + "  \n"
        report = report + "\n  Evaluation time: " + str(int(np.sum(inference_time))) + " ms  "
        App.show_message("Results", report)

    def inception_v3():
        """Inception v3 example using the hardware co-processor Neural Compute Stick (NCS) by Intel Movidius"""
        graph_file  = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/tensorflow/inception_v3/graph"
        labels_file = "/home/jorge/workspace/ncsdk/ncsdk-2.05.00.02/examples/tensorflow/inception_v3/categories.txt"

        mean        = 128
        std         = 1/128

        # Show and resize the image prior to processing
        image_name = App.ask_filename()
        Input_Image.show_image(image_name)
        image       = Input_Image.preprocess(image_name, (299, 299))
        # Image normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean) * std

        labels = Input_Image.load_labels(labels_file)
        # Run neural network using the hardware co-processor NCS
        output, inference_time = mv.manage_NCS(graph_file, image)
        order = output.argsort()[::-1][:6]
        # Report results to user
        report = "Inception v.3 results for: " + image_name + "\n\n"
        for i in range(3):
            report = report + "(" + str(i + 1) + ") " + str(round(output[order[i]] * 100, 1)) + "% " + "is some sort of " + \
                     labels[order[i]] + "  \n"
        report = report + "\n  Evaluation time: " + str(int(np.sum(inference_time))) + " ms  "
        App.show_message("Results", report)

    def batch_classifier():
        report = rapid_image_classifier.main()
        App.show_message("Results", report, 900)

    def face_detection():
        model    = "/home/jorge/workspace/OpenCV_projects/612-deep-learning-face-detection/res10_300x300_ssd_iter_140000.caffemodel"
        prototxt = "/home/jorge/workspace/OpenCV_projects/612-deep-learning-face-detection/deploy.prototxt.txt"
        dim      = (300, 300)
        confidence_treshold = 0.5
        scale_factor = 1.0

        # Load serialized model from disk
        net      = cv2.dnn.readNetFromCaffe(prototxt, model)

        # Load input image and construct input blob resizing to 300x300 pixels and normalizing
        image_name = App.ask_filename()
        image      = cv2.imread(image_name)
        (h, w)     = image.shape[:2]

        if w > MAX_WIDTH:
            scale_factor = MAX_WIDTH / w
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h))

        new_image  = cv2.resize(image, dim)
        blob       = cv2.dnn.blobFromImage(new_image, 1.0, dim, (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        # ***************************************************************
        for i in range(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections usign `confidence` greater than the minimum confidence
            if confidence > confidence_treshold:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = int(startX * scale_factor)
                startY = int(startY * scale_factor)
                endX   = int(endX   * scale_factor)
                endY   = int(endY   * scale_factor)

                # draw the bounding box of the face along with the associated probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def face_recognition():
        # Load the known faces and embeddings
        # ***************************************************************
        data = pickle.loads(open("/home/jorge/workspace/OpenCV_projects/face_recognition/encodings.pickle", "rb").read())
        detection_method = "hog" # choice of 'hog' or 'cnn'
        names = []  # list of detected names

        # Load input image and construct input blob resizing to 300x300 pixels and normalizing
        image_name = App.ask_filename()
        image    = cv2.imread(image_name)
        (h, w)   = image.shape[:2]

        # Prepare to scale image for display purposes
        # trim image if necessary
        if w > MAX_WIDTH:
            image = imutils.resize(image, width=MAX_WIDTH)
            (new_height, new_width) = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # ******************************************************************************
        # Detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings for each face
        # ******************************************************************************
        boxes     = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        # ***************************************************************
        # loop over the facial embeddings
        # ***************************************************************
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        # ***************************************************************
        # loop over the recognized faces
        # ***************************************************************
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # ***************************************************************
        # show the output image
        # ***************************************************************
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def vgg16():
        weights_file = '/home/jorge/workspace/TensorFlow_Projects/VGG16/vgg16_weights.npz'
        image_name   = App.ask_filename()
        img1         = Input_Image.show_image(image_name)
        warning_msg  = "Please be patient, the process may take up to 5 minutes..."
        messagebox.showwarning("Notice:", warning_msg)

        start  = time.time()
        session= tf.Session()
        imgs   = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg    = vgg16.vgg16(imgs, weights_file, session)
        img1   = cv2.imread(image_name)
        img1   = cv2.resize(img1, (224, 224))

        prob   = session.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds  = (np.argsort(prob)[::-1])[0:5]
        end    = time.time()
        report = "TensorFlow with VGG16 results for: " + \
                 image_name + "\n\n" + "This image contains some sort of:\n"
        for p in preds:
            report = report + "- " + vgg16.class_names[p] + \
                     " with confidence level of " + str(int(round(prob[p]*100, 1))) + "%\n"

        report = report + "\n  Evaluation time: " + str(round((end-start), 1)) + " seconds  "
        App.show_message("Results", report)

    def ssd_pre_process_image(image_name):
        dim = (300, 300)
        scale = 0.00789
        mean = [127.5, 127.5, 127.5]
        input_image = skimage.io.imread(image_name)
        colormode = "bgr"
        test_image = "/home/jorge/Pictures/dog.jpg"
        #original_image = input_image

        # Resize image to size defined during training
        tensor = skimage.transform.resize(input_image, dim, preserve_range=True)

        # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
        if (colormode == "bgr"):
            tensor = tensor[:, :, ::-1]

        # Mean subtraction & scaling
        tensor = tensor.astype(np.float32)
        tensor = (tensor - np.float32(mean)) * scale

        return tensor

    def ssd_infer_image(tensor, image_name):
        CONFIDENCE_THRESHOLD = 0.60  # 60% confident
        BASE_FILE   = "/home/jorge/workspace/ncappzoo/caffe/SSD_MobileNet/"
        graph_file  = BASE_FILE + "graph"
        labels_file = BASE_FILE + "labels.txt"

        # initialize the list of class labels our network was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ("background", "aeroplane", "bicycle", "bird",
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
        #COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        labels = [line.rstrip('\n') for line in
                  open(labels_file) if line != 'classes\n']


        original_image = skimage.io.imread(image_name)
        output, inference_time = mv.manage_NCS(graph_file, tensor)
        output_dict = deserialize_output.ssd(output, CONFIDENCE_THRESHOLD, original_image.shape)

        # Compile results
        results = "SSD Object Detection results:\n\n This image contains:\n"

        for i in range(0, output_dict['num_detections']):
            detected_class = labels[int(output_dict['detection_classes_' + str(i)])][3:]
            results = results + str(output_dict['detection_scores_' + str(i)]) + "% confidence it could be a " + \
                      detected_class + "\n"

            # Draw bounding boxes around valid detections
            (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
            (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

            # Prep string to overlay on the image
            display_str = (labels[output_dict.get('detection_classes_' + str(i))]
                           + str(output_dict.get('detection_scores_' + str(i))) + "%")

            #color = [int(c) for c in COLORS[detected_class]]
            original_image = visualize_output.draw_bounding_box(y1, x1, y2, x2,
                                                                original_image,
                                                                thickness=6,
                                                                color=(255, 255, 0),
                                                                display_str=display_str)
        results = results + "\n\nExecution time: " + str(int(np.sum(inference_time))) + " ms"

        return original_image, results

    def ssd_object_detector():
        image_name     = App.ask_filename()
        tensor         = Input_Image.ssd_pre_process_image(image_name)
        image, results = Input_Image.ssd_infer_image(tensor, image_name)
        image          = Image.fromarray(image, 'RGB')
        Input_Image.image_resize(image)
        App.show_message("SSD Results", results)

    def age_estimator():
        def execute_graph(graph_file, img):
            device = mv.open_ncs_device()
            graph = mv.load_graph(device, graph_file)
            output, inference_time = mv.infer_image(graph, img)
            input_fifo.destroy()
            output_fifo.destroy()
            graph.destroy()
            device.close()
            device.destroy()
            return output, inference_time

        # open the network graph files
        graph_file = "/home/jorge/workspace/ncappzoo/caffe/AgeNet/graph"
        mean_file  = "/home/jorge/workspace/ncappzoo/data/age_gender/age_gender_mean.npy"

        # categories for age and gender
        age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

        # read in and pre-process the image:
        ilsvrc_mean = np.load(mean_file).mean(1).mean(1)  # loading the mean file
        dim = (227, 227)

        image_name = App.ask_filename()
        Input_Image.show_image(image_name, kind="PIL", title=image_name)

        img = cv2.imread(image_name)
        img = cv2.resize(img, dim)
        img = img.astype(np.float32)
        img[:, :, 0] = (img[:, :, 0] - ilsvrc_mean[0])
        img[:, :, 1] = (img[:, :, 1] - ilsvrc_mean[1])
        img[:, :, 2] = (img[:, :, 2] - ilsvrc_mean[2])

        device = mv.open_ncs_device()
        graph  = mv.load_graph(device, graph_file)
        output, inference_time = mv.infer_image(graph, img)
        mv.close_ncs_device(device, graph)

        order     = output.argsort()
        last      = len(order) - 1
        predicted = int(order[last])
        results   = "the age range is " + age_list[predicted] + " with confidence of " + \
                    str(int((100.0 * output[predicted]))) + "%\n Time for inference: " + \
                    str(int(np.sum(inference_time))) + " ms  "
        App.show_message("Age Estimator Results", results)

    def gender_estimator():
        report = gender_estimator.main()
        App.show_message("Results", report)

    def video_gender_estimator():
        live_image_classifier.main()

    def saliency_regions():

        max_detections = 15

        # load the input image
        image_name = App.ask_filename()
        image      = cv2.imread(image_name)
        image      = Input_Image.image_resize(image, "CV2", title="Saliency Demonstration")

        # initialize OpenCV's objectness saliency detector and set the path to the input model files
        saliency = cv2.saliency.ObjectnessBING_create()
        saliency.setTrainingPath("/home/jorge/workspace/OpenCV_projects/saliency-detection/objectness_trained_model")

        # compute the bounding box predictions used to indicate saliency
        (success, saliencyMap) = saliency.computeSaliency(image)
        numDetections          = saliencyMap.shape[0]

        # loop over the detections
        for i in range(0, min(numDetections, max_detections)):
            # extract the bounding box coordinates
            (startX, startY, endX, endY) = saliencyMap[i].flatten()

            # randomly generate a color for the object and draw it on the image
            output = image.copy()
            color  = np.random.randint(0, 255, size=(3,))
            color  = [int(c) for c in color]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

            # show the output image
            cv2.imshow("Processed Image", output)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def static_saliency():
        # load the input image
        image_name   = App.ask_filename()
        image        = cv2.imread(image_name)

        # initialize OpenCV's static saliency spectral residual detector and compute the saliency map
        saliency               = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap            = (saliencyMap * 255).astype("uint8")

        # initialize OpenCV's static fine grained saliency detector and compute the saliency map
        saliency               = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(image)

        '''
        If we would like a *binary* map that we could process for contours, compute convex hull's, 
        extract bounding boxes, etc., we can additionally threshold the saliency map.
        '''
        threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # show the images
        Input_Image.image_resize(image,       "CV2", "Original Image")
        Input_Image.image_resize(saliencyMap, "CV2", "Saliency Map")
        Input_Image.image_resize(threshMap,   "CV2", "Thresh Map")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def motion_saliency():
        # initialize the motion saliency object and start the video stream
        saliency = None
        vs       = VideoStream(src=1).start()  # Using USB videocamera
        time.sleep(2.0)

        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream and resize it to 500 pixels (to speedup processing)
            frame = vs.read()
            frame = imutils.resize(frame, width=500)

            # if our saliency object is None, we need to instantiate it
            if saliency is None:
                saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                saliency.setImagesize(frame.shape[1], frame.shape[0])
                saliency.init()

            # convert the input frame to grayscale and compute the saliency map based on the motion model
            gray                   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (success, saliencyMap) = saliency.computeSaliency(gray)
            saliencyMap            = (saliencyMap * 255).astype("uint8")

            # display the image
            cv2.imshow("Frame", frame)
            cv2.imshow("Map", saliencyMap)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def pi_surveillance():
        """
        Create windows for the images from the Raspberry Pi Computers then process the images with SSD
        before presenting them on the screen.
        """
        App.create_pi_windows(650, 570)
        while True:
            A_image = Input_Image.ssd_pre_process_image(Alpha_image)
            A_image, results = Input_Image.ssd_infer_image(A_image, Alpha_image)
            cv2.imshow('Alpha', A_image)
            '''
            B_image, results = ssd.main(Beta_image)
            cv2.imshow('Beta', B_image)

            C_image, results = ssd.main(Charlie_image)
            cv2.imshow('Charlie', C_image)

            D_image, results = ssd.main(Delta_image)
            cv2.imshow('Delta', D_image)
            '''


            B_image = cv2.imread(Beta_image)
            cv2.imshow('Beta', B_image)

            C_image = cv2.imread(Charlie_image)
            cv2.imshow('Charlie', C_image)

            D_image = cv2.imread(Delta_image)
            cv2.imshow('Delta', D_image)


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

class App(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master, class_='App')
        self.filepaths = []
        main_window.geometry("700x150")
        main_window.title("ACE Demo v.3")

    def request_image():
        name = App.ask_filename()
        Input_Image.show_image(name)

    def ask_filename(starting_point=default_directory, window_title="Select Image", extension=default_file_extension):
        """
        Presents user with a selection GUI to choose a file.
        :param title: Select file window title
        :param filetypes: types of files to be displayed
        :return: name of file chosen, or name of default image.
        """
        filename = askopenfilename(initialdir=starting_point, title=window_title,
                                     filetypes=(("", extension), ("all files", "*.*")))
        if filename:
            filepaths.append(filename)
            lbl = tk.Label(main_window, text = filename)
            lbl.pack()
            return filename
        else:
            return default_image

    def compile_image_list():
        image_folder = filedialog.askdirectory(title="Choose a directory",
                                               initialdir=os.path.dirname(default_directory),
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

    # *****
    def modality(self):
        choice_list = ("HW test", "Image Classification", "Object Detection", "Surveillance",
                       "Facial Recognition", "Quit")
        modality_choice = tk.StringVar()
        modality_choice.set("Select")  # initial value

        selection_message = Message(main_window, text="Choose mode:", width=180)
        selection_message.grid(row=0, column=0, columnspan=6)

        modality_options  = OptionMenu(main_window, modality_choice, *choice_list)
        modality_options.grid(row=1, column=2)

        def mode_selection():
           choice = modality_choice.get()

           if choice   == 'HW test':
               hw_test_menu(self)
           elif choice == 'Image Classification':
               classification_menu(self)
           elif choice == 'Object Detection':
               detection_menu(self)
           elif choice == 'Surveillance':
               surveillance_menu(self)
           elif choice == 'Facial Recognition':
               facial_recognition_menu(self)
           elif choice == 'Quit':
               quit()
           else:
               messagebox.showwarning("Warning!", "Choice not available")

        modality_button = Button(main_window, text="Confirm", command=mode_selection,
                                 cursor="hand2", bg='lightblue', activebackground='yellow',
                                 activeforeground='blue', bd=4)
        modality_button.grid(row=2, column=2)
    # *****
    def hw_test_menu(menu):
        def About_hw_test():
            message = "The Hardware test menu provides choices of a variety of tests the user can " \
                      "perform to make sure the computer is ready to do the inference work.\n" \
                      "The Neural Compute Device option verifies and tests the presence of " \
                      "Neural Compute Stick (NCS) devices, which are prone to be left in a " \
                      "stale state if not closed properly, typically because of an bad file read as input.\n" \
                      "The Raspberry Pi Tests option provides four windows to verify that the Raspberry Pi " \
                      "microcomputers are sending data to the server computer.\n" \
                      "The Display Image option asks the user to select an image file and then presents " \
                      "it to the user for inspection. This can be useful to ensure an image is not " \
                      "corrupted and will work with the inference machine."
            App.show_message("About Hardware Tests", message)

        test_menu = Menu(menu)
        menu.add_cascade(     label="Tests",                    menu=test_menu)
        test_menu.add_command(label="Neural Compute Device", command=App.test_ncs)
        test_menu.add_command(label="Raspberry Pi Tests",    command=App.test_pi)
        test_menu.add_command(label="Display Image",         command=App.request_image)
        test_menu.add_command(label="About...",              command=About_hw_test)
        test_menu.add_command(label="Exit",                  command=quit)

    def classification_menu(menu):
        def About_Classification():
            message = "Classification attempts to find the most relevant element in an image and " \
                      "determine what kind of thing it is.  It does not indicate the location of " \
                      "the main object only the most likely hypothesis along with a few others " \
                      "if available.\n\n" \
                      "TensorFlow is a high level language to manipulate high order matrices for " \
                      "linear algebra, which forms the basis for neural network processing.  " \
                      "As indicated in the first TensorFlow choice, it was trained by " \
                      "Jorge Buenfil to recognize pistols, rifles, knives, bullets, and people " \
                      "(which are likely subjects of interest in a security application).  " \
                      "Additionally the Convolutional Neural Network (CNN) was taught to recognize " \
                      "that it does not know when presented with some image it cannot identify.  " \
                      "This is what is called a 'closed universe' application since everything " \
                      "the inference engine know has to fit into one of those categories.\n" \
                      "The second TensorFlow option in this menu was not trained for any particular " \
                      "security purposes, it uses a generic  VGG16 model which can recognize " \
                      "1,000 different classes of things using a 16-layer deep " \
                      "convolutional network. VGG16 achieves 92.7% top-5 test accuracy in ImageNet , " \
                      "which is a dataset of over 14 million images belonging to 1,000 classes.\n" \
                      "The next choice in the menu (AlexNet) uses the CNN architecture that won the 2012 " \
                      "ImageNet LSVRC-2012 competition by introducing some innovations that are " \
                      "still widely used today.  Namely 'RELU' and 'dropout', the first is a technique to " \
                      "accelerate up to 6 times the execution of the network and the other is a way " \
                      "to avoid overfitting (memorization of the training samples.\n" \
                      "The next choice in the menu is GoogLeNet, which is an architecture based on the " \
                      "classic LeNet network (1989) by perhaps the most widely recognized creator of neural " \
                      "networks, Yan LeCun. GoogLeNet is a deep learning algorithm with 22 layers.\n" \
                      "Inception is another deep convolutional neural network that established " \
                      "the state of the art back in 2014 at the ImageNet Large-Scale Visual Recognition " \
                      "Challenge (ILSVRC14).\n" \
                      "The next menu choice, SqueezeNet, is a network created with the goal of having " \
                      "CNNs fit into portable devices, such as cellphones, which have significantly less " \
                      "computing capabilities than desktop computers and laptops. Even with limited " \
                      "resources SqueezeNet has proven to achieve a top-1 ImageNet accuracy of 57.5% " \
                      "(meaning 57.5% of the time it was right on its first choice, and 80.3% on " \
                      "the first 5 choices (top-5 ImageNet Accuracy).\n" \
                      ""
            App.show_message("About Classification", message)

        classification_menu = Menu(menu)
        menu.add_cascade(               label  ="Classification",
                                        menu   =classification_menu)
        classification_menu.add_command(label  ="TensorFlow (trained) single image",
                                        command=Input_Image.handle_tensorflow_single)
        classification_menu.add_command(label  ="TensorFlow (trained) whole directory",
                                        command=Input_Image.handle_tensorflow_batch)
        classification_menu.add_command(label  ="TensorFlow with VGG16",
                                        command=Input_Image.vgg16)
        classification_menu.add_command(label  ="Alexnet",
                                        command=Input_Image.alexnet)
        classification_menu.add_command(label  ="GoogLeNet",
                                        command=Input_Image.googlenet)
        classification_menu.add_command(label  ="SqueezeNet",
                                        command=Input_Image.squeezenet)
        classification_menu.add_command(label  ="Triple Net Choice",
                                        command=Input_Image.triple_classification)
        classification_menu.add_command(label  ="Inception v.1",
                                        command=Input_Image.inception_v1)
        classification_menu.add_command(label  ="Inception v.3",
                                        command=Input_Image.inception_v3)
        classification_menu.add_command(label  ="Batch Classifier example",
                                        command=Input_Image.batch_classifier)
        classification_menu.add_command(label  ="Batch Classifier to CSV",
                                        command=log_image_classifier.main)
        classification_menu.add_command(label  ="About...",
                                        command=About_Classification)
        classification_menu.add_command(label  ="Exit",
                                        command=quit)

    def detection_menu(menu):
        def About_Detection():
            message = "Detection attempts to find all relevant elements in an image and " \
                      "determine what kind of thing they are.  It also indicates the location of the " \
                      "detected elements and presents them on the image encased in boxes for easier " \
                      "identification. For every object identified a confidence level for the identification" \
                      "is also presented."
            App.show_message("About Detection", message)

        detection_menu = Menu(menu)
        menu.add_cascade(          label="Detection",   menu=detection_menu)
        detection_menu.add_command(label="YOLO",     command=Input_Image.yolo)
        detection_menu.add_command(label="Single Shot Detector (SSD)", \
                                 command=Input_Image.ssd_object_detector)
        detection_menu.add_command(label="Video Object Detection", \
                                 command=video_object_detector.main)
        detection_menu.add_command(label="Video Camera Object Detection",\
                                 command=security_cam.main)
        detection_menu.add_command(label="About...", command=About_Detection)
        detection_menu.add_command(label="Exit",     command=quit)

    def facial_recognition_menu(menu):
        def About_Facial_Recognition():
            message = "Facial Recognition attempts to find human faces in an image and " \
                      "determine their location. Detected faces will be shown encased " \
                      "in boxes.  Some of the demonstration functions try to identify " \
                      "the person by face, others attempt to determine the person's age " \
                      "or gender. For every characteristic identified a confidence " \
                      "level for the identification is also presented."
            App.show_message("About Detection", message)

        face_recognition_menu = Menu(menu)
        menu.add_cascade(                 label  ="Face Recognition",
                                          menu   =face_recognition_menu)
        face_recognition_menu.add_command(label  ="Build Dataset",
                                          command=build_face_dataset.main)
        face_recognition_menu.add_command(label="Face Detection",
                                          command=Input_Image.face_detection)
        face_recognition_menu.add_command(label  ="Age Estimation",
                                          command=Input_Image.age_estimator)
        face_recognition_menu.add_command(label  ="Gender Estimation",
                                          command=Input_Image.gender_estimator)
        face_recognition_menu.add_command(  label="Video Gender Estimation",
                                          command=Input_Image.video_gender_estimator)
        face_recognition_menu.add_command(label  ="Photo Face Recognition",
                                          command=Input_Image.face_recognition)
        face_recognition_menu.add_command(label  ="Video Face Recognition",
                                          command=videofaces.main)
        face_recognition_menu.add_command(  label="About...",
                                          command=About_Facial_Recognition)
        face_recognition_menu.add_command(  label="Exit",
                                          command=quit)

    def surveillance_menu(menu):
        def About_surveillance():
            message = "The surveillance options present different choices to " \
                      "observe an area of interest and scan for a number of " \
                      "items that would trigger an action when found.  Examples " \
                      "are: pistols, rifles, knives, bullets."
            App.show_message("About Surveillance", message)

        surveillance_menu = Menu(menu)
        menu.add_cascade(             label  ="Surveillance",menu=surveillance_menu)
        surveillance_menu.add_command(label  ="Pi Surveillance", \
                                      command=Input_Image.pi_surveillance)
        surveillance_menu.add_command(label  ="Video Surveillance", \
                                      command=styg.main)
        surveillance_menu.add_command(label  ="About...", command=About_surveillance)
        surveillance_menu.add_command(label  ="Exit",     command=quit)

    def enhanced_reality_menu(menu):
        def About_Enhanced_Reality():
            message = "The Enhanced Reality menu present different ways to " \
                      "visualize how the computer processes images to obtain " \
                      "'computer vision'\n." \
                      "The Saliency Regions option scans the picture for regions of interest (use the " \
                      "space bar to see the different regions. " \
                      "As the computer analyzes each region, it determines if a known object is present " \
                      "in that  region, then puts a box on the detected objects if detection is selected, " \
                      "or simply indicates what is the dominant object in the picture if classification " \
                      "is selected.\n" \
                      "The Static Saliency option shows how the computer interprets an original image to " \
                      "identify what is important, or salient, in the picture.  " \
                      "This technique can be used to reduce the number of objects that need to be scanned " \
                      "and analyzed by the convolutional neural networks which results in time gains. " \
                      "The original image is shown along with two others, the Saliency Map highlights " \
                      "the most prominent areas according to the entropy levels detected. " \
                      "The other image, the Thresh Map, shows a further reduction of detail, based " \
                      "on the Saliency Map, but highlighting the edges of the most salient regions.\n" \
                      "The Motion Saliency option expands the notions of Saliency and Thresh Maps to " \
                      "full motion video."
            App.show_message("About Enhanced Reality", message)

        enhanced_reality_menu = Menu(menu)
        menu.add_cascade(                 label="Enhanced Reality",   menu=enhanced_reality_menu)
        enhanced_reality_menu.add_command(label="Saliency Regions",command=Input_Image.saliency_regions)
        enhanced_reality_menu.add_command(label="Static Saliency", command=Input_Image.static_saliency)
        enhanced_reality_menu.add_command(label="Motion Saliency", command=Input_Image.motion_saliency)
        enhanced_reality_menu.add_command(label="About...",        command=About_Enhanced_Reality)
        enhanced_reality_menu.add_command(label="Exit",            command=quit)

    def set_all_menus(self):
        menu = Menu(main_window)
        main_window.config(menu=menu)

        self.hw_test_menu(menu)
        self.enhanced_reality_menu(menu)
        self.classification_menu(menu)
        self.detection_menu(menu)
        self.facial_recognition_menu(menu)
        self.surveillance_menu(menu)
        #self.About()

    def show_message(title, message, width=600):
        message_window = tk.Tk()
        message_window.title(title)
        msg = tk.Message(message_window, text=message, width=width)
        msg.config(bg='lightgreen', font=('times', 12, 'italic'))
        msg.pack()

    def OpenFile():
        name = askopenfilename()
        print(name)

    def About():
        message = "Jorge Buenfil's PhD Dissertation Demo. v.3"
        App.show_message("About this application", message)

    def create_pi_windows(xmax=487, ymax=427):
        cv2.namedWindow('Alpha')
        cv2.moveWindow( 'Alpha', 0, 0)
        cv2.namedWindow('Beta')
        cv2.moveWindow( 'Beta', xmax, 0)
        cv2.namedWindow('Charlie')
        cv2.moveWindow( 'Charlie', 0, ymax)
        cv2.namedWindow('Delta')
        cv2.moveWindow( 'Delta', xmax, ymax)

    def test_pi():
        """
        Create windows for the images
        :return: None.
        """
        App.create_pi_windows()
        while (True):
            Input_Image.show_image(Alpha_image,   kind="CV2", title="Alpha")
            Input_Image.show_image(Beta_image,    kind="CV2", title="Beta")
            Input_Image.show_image(Charlie_image, kind="CV2", title="Charlie")
            Input_Image.show_image(Delta_image,   kind="CV2", title="Delta")

            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    def test_ncs():
        message = mv.test_all_ncs_devices()
        messagebox.showinfo("Test Results", message)

def main():
    app = App(main_window)

    #App.modality(App)
    App.set_all_menus(App)
    app.mainloop()

    if len(app.filepaths):
        print("selected: {}".format(app.filepaths[0]))

if __name__ == '__main__':
    main()
