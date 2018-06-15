import os
import sys
import glob
import dlib

faces_folder = "dataset/helen"

options = dlib.shape_predictor_training_options()

training_xml_path = "train.xml"
dlib.train_shape_predictor(training_xml_path, "predictor2.dat", options)
print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, "predictor2.dat")))

testing_xml_path = "test.xml"
print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, "predictor2.dat")))
