＃ - 
基于卷积神经网络的人脸识别项目

步骤1：使用take_photos.py采集人脸照片作为数据，每个人采集600张图片，保存到faceImages文件夹中

步骤2：使用get_face.py模块调用MTCNN模型进行人脸检测，将照片中的人脸保存为灰度图，放在faceImagesGray文件夹中

步骤3：使用get_data.py模块读取灰度图，并将其像素矩阵保存为数据

步骤4：使用build_model.py模块搭建CNN模型进行人脸识别，并将训练好的模型保存

步骤5：使用use_model.py模块调用摄像头拍照，并使用训练好的MTCNN和CNN模型进行人脸检测和识别
