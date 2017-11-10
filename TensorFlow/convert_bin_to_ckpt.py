#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
from yolo_model import TINY_YOLO as yolo

Tiny_YOLO_CNHW =[[16,3,3,3],[32,3,3,16],[64,3,3,32],[128,3,3,64],[256,3,3,128],[512,3,3,256],[1024,3,3,512],[1024,3,3,1024],[125,1,1,1024]]

with tf.Graph().as_default():
  feed={}
  with tf.Session() as sess:
    clf = yolo()
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
    saver.save(sess, "models/original-model.ckpt")

