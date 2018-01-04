import numpy as np
import os 
from net_description import nets 
import tensorflow as tf
import sys

sys.path.insert(0,'..')
from yolo_model import YOLO


class ConvertWeights:

    def __init__(self,nets,tiny_yolo_weights_path,yolov2_weights_path):
        self.nets = nets
        self.tiny_yolo_weights_path=os.path.basename(tiny_yolo_weights_path)
        self.yolov2_weights_path=os.path.basename(yolov2_weights_path)
        self.nets['tiny_yolo']['file']=self.tiny_yolo_weights_path
        self.nets['yolov2']['file']=self.yolov2_weights_path
        self.eps=1e-9
    
    def save_ckpt(self,key,w,b):
        print 'Creating TensorFlow Ckpt for:',key
        _type = self.nets[key]['type'] 
        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_in = tf.placeholder(tf.float32,[None,416,416,3])
                clf = YOLO(img_in,yolotype=_type)
                for i in xrange(len(clf.weights)):
                    sess.run(clf.weights[i].assign(w[i]))
                    sess.run(clf.biases[i].assign(b[i]))
                
                saver = tf.train.Saver()
                saver.save(sess,'ckpt/'+key)
                sess.close()
        tf.reset_default_graph()

    def convert(self,key):
        net = self.nets[key]
        weight_file = net['file']
        #read numpy file 
        orig=np.fromfile(weight_file, dtype=np.float32)
        #header offset in the weights file
        pos=4
        weights=[]
        biases=[]
        for shape in net['shapes'][:-1]:
              dnshape=shape[::-1]
              offset=np.product(dnshape)+4*shape[3]
              block = orig[pos:pos+offset]
              beta =     block[0:shape[3]]
              gammas =   block[1*shape[3]:2*shape[3]]
              mean =     block[2*shape[3]:3*shape[3]]
              variance = block[3*shape[3]:4*shape[3]]
              w =  np.transpose(block[4*shape[3]:].reshape(dnshape),(2,3,1,0))
              w_new=gammas[:]*w[:,:,:,:]/np.sqrt(variance[:]+self.eps)
              b_new=-gammas[:]*mean[:]/np.sqrt(variance[:]+self.eps)+beta[:]

              weights.append(w_new)
              biases.append(b_new)
              pos+=offset

        layers = net['layers']
        lastlayer = net['shapes'][layers-1]
        dnshape=lastlayer[::-1]
        offset=np.product(dnshape)+lastlayer[3]
        block = orig[pos:]
        if(len(block) != offset):
              assert False,'Format Error: Darknet weights might have been updated'
        b = block[:lastlayer[3]]
        w = np.transpose(block[lastlayer[3]:].reshape(dnshape),(2,3,1,0)).copy()
        weights.append(w)
        biases.append(b)
        self.save_ckpt(key,weights,biases)

    def run(self):
        #download weight files from darknet website
        if not os.path.exists(os.path.basename(self.tiny_yolo_weights_path)):
            assert os.system('wget '+tiny_yolo_path)==0, 'Failed to download Tiny YOLO VOC 2007 darknet weight file'
        if not os.path.exists(os.path.basename(self.yolov2_weights_path)):
            assert os.system('wget '+yolov2_path)==0, 'Failed to download YOLOv2 VOC 2007 darknet weight file'
        
        for key in self.nets.keys():
            self.convert(key)

if __name__=="__main__":

    #weight file paths on the darknet website
    tiny_yolo_path = 'https://pjreddie.com/media/files/tiny-yolo-voc.weights'
    yolov2_path ='https://pjreddie.com/media/files/yolo-voc.weights'
    os.system('mkdir ckpt')
    obj=ConvertWeights(nets,tiny_yolo_path,yolov2_path)
    obj.run()

