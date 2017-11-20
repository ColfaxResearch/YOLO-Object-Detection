#!/opt/intel/intelpython2/bin/python
import numpy as np
import tensorflow as tf
#from yolo_model import TINY_YOLO as yolo
import cv2
from matplotlib import pyplot as plt
import time
import os, sys

color={"aeroplane":(0,0,255), "bicycle":(0,255,0), "bird":(127,127,127), "boat":(127,0,127), "bottle":(127,127,0),"bus":(0,127,127), "car":(0,255,0), "cat":(0,0,255),"chair":(180,180,0), "cow":(0,180,180), "diningtable":(180,0,180), "dog":(100,0,100), "horse":(0,0,255), "motorbike":(0,100,100), "person":(0,255,255), "pottedplant":(100,100,0), "sheep":(80,80,80), "sofa":(90,150,0), "train":(0,150,100), "tvmonitor":(100,150,0) }
    
def draw_boxes(img2,box_preds):
    boxes =box_preds['boxes']
    indices = box_preds['indices']
    class_names = box_preds['class_names']

    boxes = [boxes[i] for i in indices]
    class_names = [class_names[i] for i in indices]
    for i,b in enumerate(boxes):
        left = int(max(0,b[0]))
        bot  = int(max(0,b[1]))
        right= int(min(415,b[2]))
        top  = int(min(415,b[3]))
        name = class_names[i]
        clr = color[name]
        cv2.rectangle(img2,(left,bot),(right,top),clr,2)
        cv2.putText(img2, name, (int(left), int(bot)+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2);





with tf.Graph().as_default():
    
    
    feed={}

    sess_par = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))
    
    print cv2.__version__
    
    #cam = cv2.VideoCapture(0)     
    #cam = cv2.VideoCapture('road.avi')     
    videofile = sys.argv[1]
    cam=cv2.VideoCapture(videofile)
    #Nframes = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    with sess_par as sess:
#        clf = yolo(sess)
        fps = 0
        count = 0
        time_sum = 0.0
        frame_count =0
        while(True):

            t1=time.time()
            ret,img=cam.read()
            #frame_count+=1
            ##if frame_count==Nframes-1:
            ##    frame_count=0
            ##    #cam=cv2.VideoCapture('road.avi')
            ##    cam=cv2.VideoCapture('caltech/set00/V001.seq')
            if ret==False:
                cam=cv2.VideoCapture(videofile)
                ret,img=cam.read()
                assert ret
                 


            #img=cv2.flip(img,1)
            img2=cv2.resize(img,(416,416))
            #image = np.zeros((416,416,3),dtype='float32')
            image = (img2*0.003921569)

            t2=time.time()
            feed=clf.create_feed_dict([image])
            box_preds=sess.run(clf.preds,feed)

            t3=time.time()
            draw_boxes(img2,box_preds)
            img2=cv2.resize(img2,(312,312))
            #cv2.putText(img2, '%.1f fps' % (fps), (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2);
            #cv2.imwrite('output.png',img2)
            t4=time.time()
            time_sum+=t4-t1
            #time_sum+=t3-t2
            count+=1
            if time_sum > 0.5:
              fps=count/(time_sum)
              time_sum=0
              count = 0  

#            print " "
#            print "Image load time : "+str(t2-t1)+" seconds"
#            print "YOLO time       : "+str(t3-t2)+" seconds"
#            print "Image out time  : "+str(t4-t3)+" seconds"
#            print "----------------------------------------"
#            print "Total time      : "+str(t4-t1)+" seconds"
#            print " "



