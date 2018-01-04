# YOLO-Object-Detection optimization on Xeon scalable processors  
  
In this project, optimization of TensorFlow code is performed for an object detection application to obtain real-time performance.  
Please refer the following paper for all the details regarding performance optimizations,  
https://colfaxresearch.com/yolo-optimization/  

Rquirements:  
------------
Numpy  
Python 2.7  
Tensroflow   
OpenCV  


Steps to use this code:  
----------------------

1) Go to utils/ and run:   
   $ python config.py   
   this downloads the darknet weigh files. Also, fuses batchnorm layers and creates TensorFlow Ckpt files.  

2) To run image inference:  
   $ python inference.py ,       to run TinyYolo model  
   $ python inference.py --image= [image path]  
   $ python infernce.py --v2 ,   to run YoloV2 model  
   $ NUM_INTER_THREADS=2 NUM_INTRA_THREADS=8 python inference.py  --par ,    to run parallel TensorFlow session(Inter/Intra op threads), if it is supported in your system.  
  
3) To run Webcam inference:  
   $ python webcam_inference.py    
  
  
  
Please refer the paper mentioned above to know more about system used for testing and versions of the software tools used.  

  
