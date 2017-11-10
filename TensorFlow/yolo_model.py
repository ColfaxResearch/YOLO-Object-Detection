import tensorflow as tf 
import numpy as np
from abc import ABCMeta, abstractmethod

CNHW =[[16,3,3,3],[32,3,3,16],[64,3,3,32],[128,3,3,64],[256,3,3,128],[512,3,3,256],[1024,3,3,512],[1024,3,3,1024],[125,1,1,1024]]

anchors = np.array([1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52],dtype='float32')
cl_name=np.array(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])


#class YOLO_MODEL:
#    def __init__(self,sess=None):
#        self.add_placeholder()
#        self.box_preds=self.convlayers()
#        self.preds = self.get_boxes(self.box_preds)
#        if sess is not None:
#            self.load_params(sess)        
#
#    def leaky_activation(self,input_layer):
#        return tf.maximum(input_layer,tf.scalar_mul(0.1,input_layer))
#        #return tf.nn.leaky_relu(input_layer,alpha=0.1)

class TINY_YOLO:

    def __init__(self,sess=None):
        self.add_placeholder()
        self.box_preds=self.convlayers()
        self.preds = self.get_boxes(self.box_preds)

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32,[None,416,416,3])


    def create_feed_dict(self,image):
        feed_dict ={}
        feed_dict[self.input_placeholder]=image

        return feed_dict

    def leaky_activation(self,input_layer):
        return tf.maximum(input_layer,tf.scalar_mul(0.1,input_layer))
        #return tf.nn.leaky_relu(input_layer,alpha=0.1)

    def convlayers(self, trainable=False):
        with tf.name_scope('tinyyolo') as scope:
          self.weights=[]
          self.biases =[]
  
          #conv1
          with tf.name_scope('conv1') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,3,16],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[16],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.input_placeholder,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv1 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
  
          self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')
  
          with tf.name_scope('conv2') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,16,32],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[32],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool1,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv2 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
          
          with tf.name_scope('conv3') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,32,64],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool2,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv3 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          self.pool3 = tf.nn.max_pool(self.conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool3')
  
          with tf.name_scope('conv4') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,64,128],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool3,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv4 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          self.pool4 = tf.nn.max_pool(self.conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool4')
  
          with tf.name_scope('conv5') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,128,256],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool4,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv5 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          self.pool5 = tf.nn.max_pool(self.conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool5')
  
          with tf.name_scope('conv6') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,256,512],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[512],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool5,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv6 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          self.pool6 = tf.nn.max_pool(self.conv6,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool6')
  
          with tf.name_scope('conv7') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,512,1024],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.pool6,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv7 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
  
          with tf.name_scope('conv8') as scope:
              kernel=tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[1024],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.conv7,kernel,[1,1,1,1],padding='SAME')
              out = tf.nn.bias_add(conv,biases)
              self.conv8 = self.leaky_activation(out)
              self.weights +=[kernel]
              self.biases +=[biases]
          
          with tf.name_scope('conv9') as scope:
              kernel=tf.Variable(tf.truncated_normal([1,1,1024,125],dtype=tf.float32,stddev=1e-1),name='weights',trainable=trainable)
              biases = tf.Variable(tf.constant(0.0,shape=[125],dtype=tf.float32),trainable=trainable,name='biases')
              conv = tf.nn.conv2d(self.conv8,kernel,[1,1,1,1],padding='VALID')
              self.conv9 = tf.nn.bias_add(conv,biases)
              box_preds = self.conv9
              self.weights +=[kernel]
              self.biases +=[biases]
          
        return box_preds


    def get_boxes(self,box_preds):

        global anchors
        global cl_name

        bx_preds= tf.reshape(box_preds,[-1,13,13,5,25])
        self.bx_preds = bx_preds
        confs = tf.sigmoid(bx_preds[:,:,:,:,4])
        class_probs =tf.nn.softmax(bx_preds[:,:,:,:,5:25])
        max_class =tf.reduce_max(class_probs,axis=4)*confs
        max_idx =tf.argmax(class_probs,axis=4)

        indices = tf.where(max_class>0.20)
        class_names =tf.gather(cl_name,tf.gather_nd(max_idx,indices))
        scores = tf.gather_nd(max_class,indices)
        
        tx = tf.gather_nd(bx_preds[:,:,:,:,0],indices)
        ty = tf.gather_nd(bx_preds[:,:,:,:,1],indices)
        tw = tf.gather_nd(bx_preds[:,:,:,:,2],indices)
        th = tf.gather_nd(bx_preds[:,:,:,:,3],indices)
        batch_addr = indices[:,0]
        x = (tf.cast(indices[:,2],tf.float32) + tf.sigmoid(tx))*32.0
        y = (tf.cast(indices[:,1],tf.float32) + tf.sigmoid(ty))*32.0
        w = tf.exp(tw)*tf.gather(anchors,2*indices[:,3])*16.0
        h = tf.exp(th)*tf.gather(anchors,2*indices[:,3]+1)*16.0
        x1=x-w
        x2=x+w
        y1=y-h
        y2=y+h
        boxes = tf.stack([x1,y1,x2,y2],axis=1)
        indices = tf.image.non_max_suppression(boxes,scores,10,iou_threshold=0.3)
        preds={'batch_addr':batch_addr, 'boxes':boxes,'indices':indices,'class_names':class_names}
        return preds








