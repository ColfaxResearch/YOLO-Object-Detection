#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
from model import TINY_YOLO as yolo
#from model_tf_nchw import TINY_YOLO as yolo
import cv2
from matplotlib import pyplot as plt
from scipy.linalg import norm
import time
import os, sys



os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


with tf.Graph().as_default():
    
    
    feed={}

    
    #config = tf.ConfigProto()
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #sess_xla = tf.Session(config=config)
    
    cam = cv2.VideoCapture(int(sys.argv[1]))     
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))) as sess:
        clf = yolo(sess)
        fps=0.0
        count = 0
        time_sum = 0.0
        cam_fps=0.0
        cam_count = 0
        cam_time_sum = 0.0

        while(True):
            #if count > 30:
            #  break
            #count+=1
            #path='samples/dog.jpg'
            #img=cv2.imread(path)
            t1=time.time()
	    ret,img = cam.read()
            img2=cv2.resize(cv2.flip(img,1),(416,416))
            image = (img2*0.003921569)
            t2=time.time()

            feed=clf.create_feed_dict([image])
            box_preds=sess.run(clf.preds,feed)
            #box_preds=sess.run(clf.box_preds,feed)

            t3=time.time()

            boxes =box_preds['boxes']
            indices = box_preds['indices']
            class_names = box_preds['class_names']
            
            boxes = [boxes[i] for i in indices]
            class_names = [class_names[i] for i in indices]
            for i,b in enumerate(boxes):
                left = int(max(1,b[0]))
                bot  = int(max(1,b[1]))
                right= int(min(415,b[2]))
                top  = int(min(415,b[3]))
                cv2.rectangle(img2,(left,bot),(right,top),(0,255,255),2)
                cv2.putText(img2, class_names[i], (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2);
            img2=cv2.resize(img2,(312,312))
            cv2.putText(img2, '%.1f fps' % (fps), (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2);
	    cv2.imshow('webcam', img2)
	    if cv2.waitKey(1) == 27:
		break
            t4=time.time()
            time_sum+=t4-t1
            #time_sum+=t3-t2
            count+=1
            if time_sum > 0.5:
              fps=count/(time_sum)
              time_sum=0
              count = 0  

            print "--------------------------------------- "
            print "Image load time : "+str(t2-t1)+" seconds"
            print "YOLO time       : "+str(t3-t2)+" seconds"
            print "Image out time  : "+str(t4-t3)+" seconds"
            print "----------------------------------------"
            print "Total time      : "+str(t4-t1)+" seconds"
            print "----------------------------------------"

	
            #plt.imshow(img2)
            #plt.show()

