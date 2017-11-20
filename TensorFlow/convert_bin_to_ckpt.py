#!/opt/intel/intelpython2/bin/python
import sys
if(len(sys.argv) < 2):
  print("Usage: %s [yolotype]" % sys.argv[0])
  exit()
yolotype=sys.argv[1]

import numpy as np
import tensorflow as tf
from yolo_model import YOLO

Tiny_YOLO_CNHW =[[16,3,3,3],[32,3,3,16],[64,3,3,32],[128,3,3,64],[256,3,3,128],[512,3,3,256],[1024,3,3,512],[1024,3,3,1024],[125,1,1,1024]]


with tf.Graph().as_default():
  feed={}
  with tf.Session() as sess:
    if yolotype=="TINY_VOC":
      img_in = tf.placeholder(tf.float32,[None,416,416,3])
      clf = YOLO(img_in)
      print "Loading Weights.."

      for i in xrange(9):
          wfile = 'parameters/conv'+str(i+1)+'_W.bin'
          w = np.fromfile(wfile,dtype='float32')
          shape = Tiny_YOLO_CNHW[i]
          w = np.transpose(w.reshape(shape),(1,2,3,0))
          sess.run(clf.weights[i].assign(w))
          temp=sess.run(clf.weights[i])
          val = np.sum(temp-w)
          if(val!=0):
              print ">> Error loading param[weight]"+str(i)
              assert False
      
      print "Loading Biases.."
      print ""
      for i in xrange(9):
          bfile = 'parameters/conv'+str(i+1)+'_b.bin'
          b = np.fromfile(bfile,dtype='float32')
          sess.run(clf.biases[i].assign(b))
          temp=sess.run(clf.biases[i])
          val = np.sum(temp-b)
          if(val!=0):
              print ">> Error loading param[bias]"+str(i)
              assert False
      
      print ""
      print "Initialization sucessfull.."
      print ""
      print ""

      saver = tf.train.Saver()
      saver.save(sess, "models/original-tiny-voc-model.ckpt")
    elif yolotype=="V2_VOC":
      img_in = tf.placeholder(tf.float32,[None,416,416,3])
      clf = YOLO(img_in, yolotype=yolotype)
      w=np.load('yolov2/yolov2-w.npy')
      b=np.load('yolov2/yolov2-b.npy')
      for i in xrange(len(clf.weights)):
        sess.run(clf.weights[i].assign(w[i]))
        sess.run(clf.biases[i].assign(b[i]))
      saver = tf.train.Saver()
      saver.save(sess, "models/original-v2-voc-model.ckpt")

    else:
      print("Unkown yolotype: "+yolotype)

