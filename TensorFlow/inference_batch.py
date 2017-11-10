#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
from yolo_model_batch import TINY_YOLO as yolo
import cv2
from matplotlib import pyplot as plt
import time
import os, sys


    
    
    
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





with tf.Graph().as_default():
    
    
    feed={}


    
    sess_par = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))
    batch = int(sys.argv[1])

    print "*******************************************************"
    print "Batch Size : ",batch
    with sess_par as sess:
        clf = yolo(sess)
        fps = 0
        count = 0
        time_sum = 0.0
        for T in xrange(10):

            t1=time.time()
            img=cv2.imread('dog.jpg')
            img1=cv2.resize(img,(416,416))
            image = (img1*0.003921569)
            im_list = [image]*batch
            t2=time.time()
            feed=clf.create_feed_dict(im_list)
            box_preds=sess.run(clf.preds,feed)

            t3=time.time()
            draw_boxes(im_list,box_preds)
            #cv2.imshow('frame',img2)
            #if cv2.waitKey(1) == 27:
            #    break
            t4=time.time()

            print " "
            print "Image load time : "+str(t2-t1)+" seconds"
            print "YOLO time       : "+str(t3-t2)+" seconds"
            print "Image out time  : "+str(t4-t3)+" seconds"
            print "----------------------------------------"
            print "Total time      : "+str(t4-t1)+" seconds"
            print " "

    print ""
    print ""
    print ""
    print ""
