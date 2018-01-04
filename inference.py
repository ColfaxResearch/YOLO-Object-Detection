#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
from yolo_model import YOLO
import cv2
from matplotlib import pyplot as plt
import time
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Sample program for YOLO inference in TensroFlow')

parser.add_argument('--v2',action='store_true', default=False, help='Type of the yolo mode Tiny or V2')
parser.add_argument('--par',action='store_true', default=False, help='Enable parallel session with INTER/INTRA TensorFlow threads')
parser.add_argument('--image',action='store', default='sample/dog.jpg', help='Select image for object detection')

args = parser.parse_args()

#select the checkpoint
if not args.v2:
    path="utils/ckpt/tiny_yolo"
    _type = 'TINY_VOC'
else:
    path="utils/ckpt/yolov2"
    _type = 'V2_VOC'





#Helper function to draw boxes deduced from the feature map
def draw_boxes(img,box_preds):

    batch_addr = box_preds['batch_addr']
    boxes =box_preds['boxes']
    indices = box_preds['indices']
    class_names = box_preds['class_names']

    boxes = [boxes[i] for i in indices]
    class_names = [class_names[i] for i in indices]
    for i,b in enumerate(boxes):
        idx  = batch_addr[i]
        left = int(max(0,b[0]))
        bot  = int(max(0,b[1]))
        right= int(min(415,b[2]))
        top  = int(min(415,b[3]))
        cv2.rectangle(img[idx],(left,bot),(right,top),(0,255,255),2)
        cv2.putText(img[idx], class_names[i], (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2);


#TensorFlow graph and Session
with tf.Graph().as_default():

    batch = 1
    img_in = tf.placeholder(tf.float32,[None,416,416,3])
    clf = YOLO(img_in,yolotype=_type)
    saver = tf.train.Saver()

    #read and preprocess the image
    img=cv2.imread(args.image)
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img2=[cv2.resize(img1,(416,416))]*batch
    image = [im*0.003921569 for im in img2]

    #select the session Type
    if not args.par:
        sess_type = tf.Session()
    else:
        sess_type = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                                    intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))

    with sess_type as sess:
        saver.restore(sess, path)
        t0 = time.time()
        box_preds=sess.run(clf.preds,{img_in: image})
        t1 = time.time()
        print("Compute Time : %f seconds"  % (t1-t0))
        draw_boxes(img2,box_preds)
        plt.imshow(img2[0])
        plt.show()
        sess.close()
