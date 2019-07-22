import cv2
import numpy as np
import tensorflow as tf
import time
import sys
sys.dont_write_bytecode = True

from mtcnn.mtcnn import MTCNN

detector = MTCNN()

name = ['a', 'b', 'c', 'd', 'e', 'f', 
        'g', 'h', 'i', 'j']

# =============================================================================
# file_path = r'C:\Users\hasee\Desktop\face_recognition\faceImages\caimingyue\521.jpg'
# #file_path = r'C:\Users\hasee\Desktop\625494820355192462.jpg'
# 
# image = cv2.imread(file_path)
# result = detector.detect_faces(image)
# 
# for i in range(len(result)):
#     bounding_box = result[i]['box']
#     
#     for k in range(len(bounding_box)):
#         if bounding_box[k] < 0:
#             bounding_box[k] = 0
#             
#     image_face = image[bounding_box[1] : bounding_box[1] + bounding_box[3], 
#                        bounding_box[0] : bounding_box[0] + bounding_box[2], :]
#     image_face = cv2.cvtColor(image_face, cv2.COLOR_RGB2GRAY)
#     
#     pad_1 = int((np.max(image_face.shape) - np.min(image_face.shape)) / 2)
#     pad_2 = np.max(image_face.shape) - np.min(image_face.shape) - pad_1
#     if image_face.shape[0] >= image_face.shape[1]:
#         pad_mat_1 = np.uint8(np.zeros([image_face.shape[0], pad_1]))
#         pad_mat_2 = np.uint8(np.zeros([image_face.shape[0], pad_2]))
#         image_face = np.c_[pad_mat_1, image_face, pad_mat_2]
#     else:
#         pad_mat_1 = np.uint8(np.zeros([pad_1, image_face.shape[1]]))
#         pad_mat_2 = np.uint8(np.zeros([pad_2, image_face.shape[1]]))
#         image_face = np.r_[pad_mat_1, image_face, pad_mat_2]
#     
#     cv2.imshow('image_face', image_face)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     
#     image_face = cv2.resize(image_face, (32, 32))
#     image_face = np.float32(np.reshape(image_face / 255, [1, 32, 32, 1]))
#     
#     with tf.Session() as sess:
#         new_saver = tf.train.import_meta_graph('./model/My_Model.meta')
#         new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
#         
#         graph = tf.get_default_graph()
#         x_new = graph.get_tensor_by_name('x:0')
#         y_new = graph.get_tensor_by_name('y_conv:0')
#         keep_prob_new = graph.get_tensor_by_name('keep_prob:0')
#         
#         pre_y = sess.run(y_new, feed_dict = {x_new: image_face, keep_prob_new: 1.0})
#         print(name[int(np.argmax(pre_y, axis = 1))], np.max(pre_y))
# 
#     cv2.rectangle(image,
#                   (bounding_box[0], bounding_box[1]),
#                   (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#                   (0,155,255),
#                   2)
#         
#     cv2.putText(image, name[int(np.argmax(pre_y, axis = 1))] + ': ' + str(np.max(pre_y)), 
#                 (bounding_box[0], bounding_box[1] + bounding_box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
# 
# 
# cv2.imshow('face', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
# =============================================================================





cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    
    result = detector.detect_faces(image)
    
    for i in range(len(result)):
        bounding_box = result[i]['box']
        
        for k in range(len(bounding_box)):
            if bounding_box[k] < 0:
                bounding_box[k] = 0
                
        image_face = image[bounding_box[1] : bounding_box[1] + bounding_box[3], 
                           bounding_box[0] : bounding_box[0] + bounding_box[2], :]
        image_face = cv2.cvtColor(image_face, cv2.COLOR_RGB2GRAY)
        
        pad_1 = int((np.max(image_face.shape) - np.min(image_face.shape)) / 2)
        pad_2 = np.max(image_face.shape) - np.min(image_face.shape) - pad_1
        if image_face.shape[0] >= image_face.shape[1]:
            pad_mat_1 = np.uint8(np.zeros([image_face.shape[0], pad_1]))
            pad_mat_2 = np.uint8(np.zeros([image_face.shape[0], pad_2]))
            image_face = np.c_[pad_mat_1, image_face, pad_mat_2]
        else:
            pad_mat_1 = np.uint8(np.zeros([pad_1, image_face.shape[1]]))
            pad_mat_2 = np.uint8(np.zeros([pad_2, image_face.shape[1]]))
            image_face = np.r_[pad_mat_1, image_face, pad_mat_2]
        
        
        image_face = cv2.resize(image_face, (32, 32))
        image_face = np.float32(np.reshape(image_face / 255, [1, 32, 32, 1]))
        
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./model/My_Model.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            
            graph = tf.get_default_graph()
            x_new = graph.get_tensor_by_name('x:0')
            y_new = graph.get_tensor_by_name('y_conv:0')
            keep_prob_new = graph.get_tensor_by_name('keep_prob:0')
            
            pre_y = sess.run(y_new, feed_dict = {x_new: image_face, keep_prob_new: 1.0})
            print(name[int(np.argmax(pre_y, axis = 1))], np.max(pre_y))
    
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)
            
        cv2.putText(image, name[int(np.argmax(pre_y, axis = 1))] + ': ' + str(np.max(pre_y)), 
                    (bounding_box[0], bounding_box[1] + bounding_box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)



    cv2.imshow("capture", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    
    time.sleep(0.1)

    
cap.release()    
cv2.destroyAllWindows()





















