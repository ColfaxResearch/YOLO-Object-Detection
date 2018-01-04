import tensorflow as tf 
import numpy as np

###
# Supported yolo-types
#  TINY_VOC
#  V2_VOC
###
class YOLO:
  def __init__(self, input_images, yolotype='TINY_VOC'):
    #tf.input_placeholder(tf.float32,[None,416,416,3])
    self.weights =[]
    self.biases =[]
    self.trainable=False
    self.input_images = input_images
    self.yolotype=yolotype
    if self.yolotype == 'V2_VOC':
      self.num_classes = 20
      self.cl_names=np.array(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
      self.anchors = np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],dtype='float32')
      self.conv_pipe = self.yoloV2ConvPipe(self.input_images)
    elif self.yolotype == 'TINY_VOC':
      self.num_classes = 20
      self.cl_names=np.array(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
      self.anchors = np.array([1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52],dtype='float32')
      self.conv_pipe = self.tinyConvPipe(self.input_images)
    else:
      print("Unkown yolotype:")
      assert False
    self.preds = self.NMSPipe(self.conv_pipe)

  def _leakyConv(self,input_layer, shape, stride=[1,1,1,1]):
    w = tf.Variable(tf.truncated_normal(shape,dtype=tf.float32,stddev=1e-1),name='weights',trainable=self.trainable)
    b = tf.Variable(tf.constant(0.0,shape=[shape[3]],dtype=tf.float32),trainable=self.trainable,name='biases')
    conv = tf.nn.bias_add(tf.nn.conv2d(input_layer,w,stride,padding='SAME'), b)
    self.weights +=[w]
    self.biases +=[b]
    relu = tf.maximum(conv,tf.scalar_mul(0.1,conv))
    return relu

  def _linearConv(self,input_layer, shape, stride=[1,1,1,1]):
    w = tf.Variable(tf.truncated_normal(shape,dtype=tf.float32,stddev=1e-1),name='weights',trainable=self.trainable)
    b = tf.Variable(tf.constant(0.0,shape=[shape[3]],dtype=tf.float32),trainable=self.trainable,name='biases')
    conv = tf.nn.bias_add(tf.nn.conv2d(input_layer,w,stride,padding='SAME'), b)
    self.weights +=[w]
    self.biases +=[b]
    return conv

  def tinyConvPipe(self, input_images):
    with tf.name_scope('tinyyolo') as scope:
      with tf.name_scope('conv1') as scope:
        conv1 = self._leakyConv(input_images, [3,3,3,16])
  
      pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')
  
      with tf.name_scope('conv2') as scope:
        conv2 = self._leakyConv(pool1, [3,3,16,32])
      pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
          
      with tf.name_scope('conv3') as scope:
        conv3 = self._leakyConv(pool2, [3,3,32,64])
      pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool3')
  
      with tf.name_scope('conv4') as scope:
        conv4 = self._leakyConv(pool3, [3,3,64,128])
      pool4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool4')
  
      with tf.name_scope('conv5') as scope:
        conv5 = self._leakyConv(pool4, [3,3,128,256])
      pool5 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool5')
  
      with tf.name_scope('conv6') as scope:
        conv6 = self._leakyConv(pool5, [3,3,256,512])
      pool6 = tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool6')
  
      with tf.name_scope('conv7') as scope:
        conv7 = self._leakyConv(pool6, [3,3,512,1024])
  
      with tf.name_scope('conv8') as scope:
        conv8 = self._leakyConv(conv7, [3,3,1024,1024])
          
      with tf.name_scope('conv9') as scope:
        conv9 = self._linearConv(conv8, [1,1,1024,125])
          
      return conv9

  def yoloV2ConvPipe(self, input_images):
    with tf.name_scope('yolov2') as scope:
      with tf.name_scope('conv1') as scope:
        conv1 = self._leakyConv(input_images, [3,3,3,32])
      pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool1')
  
      with tf.name_scope('conv2') as scope:
        conv2 = self._leakyConv(pool1, [3,3,32,64])
      pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
 
      with tf.name_scope('conv3') as scope:
        conv3 = self._leakyConv(pool2, [3,3,64,128])
      with tf.name_scope('conv4') as scope:
        conv4 = self._leakyConv(conv3, [1,1,128,64])
      with tf.name_scope('conv5') as scope:
        conv5 = self._leakyConv(conv4, [3,3,64,128])
      pool3 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
 
      with tf.name_scope('conv6') as scope:
        conv6 = self._leakyConv(pool3, [3,3,128,256])
      with tf.name_scope('conv7') as scope:
        conv7 = self._leakyConv(conv6, [1,1,256,128])
      with tf.name_scope('conv8') as scope:
        conv8 = self._leakyConv(conv7, [3,3,128,256])
      pool4 = tf.nn.max_pool(conv8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')
 
      with tf.name_scope('conv9') as scope:
        conv9 = self._leakyConv(pool4, [3,3,256,512])
      with tf.name_scope('conv10') as scope:
        conv10 = self._leakyConv(conv9, [1,1,512,256])
      with tf.name_scope('conv11') as scope:
        conv11 = self._leakyConv(conv10, [3,3,256,512])
      with tf.name_scope('conv12') as scope:
        conv12 = self._leakyConv(conv11, [1,1,512,256])
      with tf.name_scope('conv13') as scope:
        conv13 = self._leakyConv(conv12, [3,3,256,512])
      pool5 = tf.nn.max_pool(conv13,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='pool2')

      with tf.name_scope('conv14') as scope:
        conv14 = self._leakyConv(pool5, [3,3,512,1024])
      with tf.name_scope('conv15') as scope:
        conv15 = self._leakyConv(conv14, [1,1,1024,512])
      with tf.name_scope('conv16') as scope:
        conv16 = self._leakyConv(conv15, [3,3,512,1024])
      with tf.name_scope('conv17') as scope:
        conv17 = self._leakyConv(conv16, [1,1,1024,512])
      with tf.name_scope('conv18') as scope:
        conv18 = self._leakyConv(conv17, [3,3,512,1024])
 
      with tf.name_scope('conv19') as scope:
        conv19 = self._leakyConv(conv18, [3,3,1024,1024])
      with tf.name_scope('conv20') as scope:
        conv20 = self._leakyConv(conv19, [3,3,1024,1024])

      # routes
      with tf.name_scope('conv21') as scope:
        conv21 = self._leakyConv(conv13, [1,1,512,64])
      #reorg=tf.concat([conv21[:,::2,::2,:],conv21[:,::2,1::2,:],conv21[:,1::2,::2,:],conv21[:,1::2,1::2,:]], axis=3)
      reorg=tf.reshape(conv21, [-1, 13, 2, 13, 2, 64])
      reorg=tf.transpose(reorg, [0, 1, 3, 2, 4, 5])
      reorg=tf.reshape(reorg, [-1, 13, 13, 256])
      route=tf.concat([reorg, conv20], axis=3)

      with tf.name_scope('conv22') as scope:
        conv22 = self._leakyConv(route, [3,3,1280,1024])

      with tf.name_scope('conv23') as scope:
        conv23 = self._linearConv(conv22, [1,1,1024,125])
      return conv23

  def NMSPipe(self,box_preds):

    global anchors
    global cl_name

    bx_preds= tf.reshape(box_preds,[-1,13,13,5,25])
    self.bx_preds = bx_preds
    confs = tf.sigmoid(bx_preds[:,:,:,:,4])
    class_probs =tf.nn.softmax(bx_preds[:,:,:,:,5:25])
    max_class =tf.reduce_max(class_probs,axis=4)*confs
    self.max_class=max_class
    max_idx =tf.argmax(class_probs,axis=4)

    indices = tf.where(max_class>0.20)
    class_names =tf.gather(self.cl_names,tf.gather_nd(max_idx,indices))
    scores = tf.gather_nd(max_class,indices)
    
    tx = tf.gather_nd(bx_preds[:,:,:,:,0],indices)
    ty = tf.gather_nd(bx_preds[:,:,:,:,1],indices)
    tw = tf.gather_nd(bx_preds[:,:,:,:,2],indices)
    th = tf.gather_nd(bx_preds[:,:,:,:,3],indices)
    batch_addr = indices[:,0]
    x = (tf.cast(indices[:,2],tf.float32) + tf.sigmoid(tx))*32.0
    y = (tf.cast(indices[:,1],tf.float32) + tf.sigmoid(ty))*32.0
    w = tf.exp(tw)*tf.gather(self.anchors,2*indices[:,3])*16.0
    h = tf.exp(th)*tf.gather(self.anchors,2*indices[:,3]+1)*16.0
    x1=x-w
    x2=x+w
    y1=y-h
    y2=y+h
    boxes = tf.stack([x1,y1,x2,y2],axis=1)
    indices = tf.image.non_max_suppression(boxes,scores,10,iou_threshold=0.3)
    preds={'batch_addr':batch_addr, 'boxes':boxes,'indices':indices,'class_names':class_names}
    return preds
