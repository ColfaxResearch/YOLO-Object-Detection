#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
from yolo_model import YOLO
import cv2
from matplotlib import pyplot as plt
import time
import os, sys
from scipy.misc import imread,imsave,imresize

color={"aeroplane":(0,0,255), "bicycle":(0,255,0), "bird":(127,127,127), "boat":(127,0,127), "bottle":(127,127,0),"bus":(0,127,127), "car":(0,255,0), "cat":(0,0,255),"chair":(180,180,0), "cow":(0,180,180), "diningtable":(180,0,180), "dog":(100,0,100), "horse":(0,0,255), "motorbike":(0,100,100), "person":(0,255,255), "pottedplant":(100,100,0), "sheep":(80,80,80), "sofa":(90,150,0), "train":(0,150,100), "tvmonitor":(100,150,0) }
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
  
  img_in = tf.placeholder(tf.float32,[None,416,416,3])
  clf = YOLO(img_in, 'V2_VOC')
  #clf = YOLO(img_in)
  saver = tf.train.Saver()
  img=cv2.imread(sys.argv[1])
  img2=cv2.resize(img,(416,416))
  image = [(img2*0.003921569)]*10

  with sess_par as sess:
    saver.restore(sess, "models/original-v2-voc-model.ckpt")
    #saver.restore(sess, "models/original-tiny-voc-model.ckpt")
    for i in xrange(20):
      t0 = time.time()
      box_preds=sess.run(clf.preds,{img_in: image})
      t1 = time.time()
      print("%d: %f"  % (i, t1-t0))
    draw_boxes(img2,box_preds)
    cv2.imwrite('output.jpg',img2)
    print("Output image saved to output.jpg")
