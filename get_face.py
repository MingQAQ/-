import numpy as np
import cv2
import os

import sys
sys.dont_write_bytecode = True

from mtcnn.mtcnn import MTCNN

detector = MTCNN()

file = './faceImages/'
new_file = './faceImagesGray/'

name = ['a', 'b', 'c', 'd', 'e', 'f', 
        'g', 'h', 'i', 'j']

for i in range(10):
    file_path = new_file + name[i] + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    number = 0
    for j in range(600):
        image = cv2.imread(file + name[i] + '/' + str(j) + '.jpg')
        result = detector.detect_faces(image)

        if len(result) != 0:
            bounding_box = result[0]['box']
                        
            for k in range(len(bounding_box)):
                if bounding_box[k] < 0:
                    bounding_box[k] = 0
                        
            image = image[bounding_box[1] : bounding_box[1] + bounding_box[3], 
                          bounding_box[0] : bounding_box[0] + bounding_box[2], :]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            pad_1 = int((np.max(image.shape) - np.min(image.shape)) / 2)
            pad_2 = np.max(image.shape) - np.min(image.shape) - pad_1
            if image.shape[0] >= image.shape[1]:
                pad_mat_1 = np.uint8(np.zeros([image.shape[0], pad_1]))
                pad_mat_2 = np.uint8(np.zeros([image.shape[0], pad_2]))
                image = np.c_[pad_mat_1, image, pad_mat_2]
            else:
                pad_mat_1 = np.uint8(np.zeros([pad_1, image.shape[1]]))
                pad_mat_2 = np.uint8(np.zeros([pad_2, image.shape[1]]))
                image = np.r_[pad_mat_1, image, pad_mat_2]
            
            image = cv2.resize(image, (32, 32))

            #cv2.imshow('face', image)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
                
            cv2.imwrite(file_path + str(number) + '.jpg', image)
            number += 1