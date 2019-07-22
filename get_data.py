import numpy as np
import os
import cv2

file = './faceImagesGray/'

train_data = np.zeros(shape = (0, 1024))
train_label = np.zeros(shape = (0))

test_data = np.zeros(shape = (0, 1024))
test_label = np.zeros(shape = (0))

name = ['a', 'b', 'c', 'd', 'e', 'f', 
        'g', 'h', 'i', 'j']

for i in range(10):
    single_file = file + name[i] + '/'
    images_flies = os.listdir(single_file)
    
    arr = np.arange(len(images_flies))
    np.random.shuffle(arr)
    train_arr = arr[0 : int(0.8 * len(images_flies))]
    test_arr = arr[int(0.8 * len(images_flies)) : len(images_flies)]
    
    for ii in train_arr:
        image = cv2.imread(single_file + images_flies[ii], 0) / 255
        image = image.reshape([1, 1024])
        train_data = np.vstack((train_data, image))
        train_label = np.hstack((train_label, np.array([i])))

    for ii in test_arr:
        image = cv2.imread(single_file + images_flies[ii], 0) / 255
        image = image.reshape([1, 1024])
        test_data = np.vstack((test_data, image))
        test_label = np.hstack((test_label, np.array([i])))

total_train_arr = np.arange(train_data.shape[0])
np.random.shuffle(total_train_arr)
train_data = train_data[total_train_arr]
train_label = train_label[total_train_arr]

total_test_arr = np.arange(test_data.shape[0])
np.random.shuffle(total_test_arr)
test_data = test_data[total_test_arr]
test_label = test_label[total_test_arr]

np.savez('data', train_data = train_data, test_data = test_data)
np.savez('labels', train_label = train_label, test_label = test_label)