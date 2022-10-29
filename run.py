# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
import warnings
import argparse

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy

def main(yolo,video_path):
    print(f'Using {yolo} model')
    # Definition of the parameters
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    out_dir = 'videos/output/'
    # print('The output folder is', out_dir)
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)

    all_frames = []
    # for video in args.videos:
    # loadvideo = LoadVideo(video_path)
    # video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
    # frame_nums = len(all_frames)
    # tracking_path = out_dir + 'tracking' + '.avi'
    # combined_path = out_dir + 'allVideos' + '.avi'
    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is not True:
            cap.release()
            break
        if ret:
            h, w, _ = frame.shape 
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs = yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
            features = encoder(frame, boxs)  # n * 128
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n
            text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
            # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]  # length = len(indices)

            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            tmp_ids = []
            #print("Tracks",tracker.tracks)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                    tmp_ids.append(track.track_id)
                    if track.track_id not in track_cnt:
                        track_cnt[track.track_id] = [
                            [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                        ]
                        images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                    else:
                        track_cnt[track.track_id].append([
                            frame_cnt,
                            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                            area
                        ])
                        images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                cv2_addBox(
                    track.track_id,
                    frame,
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    line_thickness,
                    text_thickness,
                    text_scale
                )
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

        
    

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    video_path = './videos/init/Single1.mp4'
    main(YOLO4(),video_path)
