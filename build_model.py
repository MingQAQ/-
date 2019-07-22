import numpy as np
import tensorflow as tf

image_data = np.load('data.npz')
image_label = np.load('labels.npz')

train_data = image_data['train_data']
train_label = image_label['train_label']
test_data = image_data['test_data']
test_label = image_label['test_label']

train_images = np.float32(np.reshape(train_data, [train_data.shape[0], 32, 32, 1]))
test_images = np.float32(np.reshape(test_data, [test_data.shape[0], 32, 32, 1]))

tf.reset_default_graph()

label = tf.placeholder(tf.int32, shape = [None], name = 'label')
labels = tf.one_hot(label, 11, name = 'labels')

x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name = 'x')
y = tf.placeholder(tf.float32, shape = [None, 11], name = 'y')

w1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 32], stddev = 0.1), name = 'w1')
b1 = tf.Variable(tf.constant(0.1, shape = [32]), name = 'b1')
h_conv1 = tf.nn.relu(tf.nn.conv2d(input = x, filter = w1, strides = [1, 1, 1, 1], padding = 'VALID') + b1, name = 'h_conv1')
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'h_pool1')

w2 = tf.Variable(tf.truncated_normal(shape = [5, 5, 32, 64], stddev = 0.1), name = 'w2')
b2 = tf.Variable(tf.constant(0.1, shape = [64]), name = 'b2')
h_conv2 = tf.nn.relu(tf.nn.conv2d(input = h_pool1, filter = w2, strides = [1, 1, 1, 1], padding = 'VALID') + b2, name = 'h_conv2')
h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'h_pool2')

w_fc1 = tf.Variable(tf.truncated_normal(shape = [5*5*64, 512], stddev = 0.1), name = 'w_fc1')
b_fc1 = tf.Variable(tf.constant(0.1, shape = [512]), name = 'b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64], name = 'h_pool2_flat')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name = 'h_fc1')

keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = 'h_fc1_drop')

w_fc2 = tf.Variable(tf.truncated_normal(shape = [512, 10], stddev = 0.1), name = 'w_fc2')
b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]), name = 'b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2, name = 'y_conv')

cross_entropy = tf.reduce_mean(-tf.reduce_mean(y * tf.log(y_conv), axis = 1), name = 'cross_entropy')

train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1), name = 'correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    train_labels = sess.run(labels, feed_dict = {label: train_label})
    test_labels = sess.run(labels, feed_dict = {label: test_label})
    
    batch_size = 50
    num_epochs = 50
    num_batch = int(train_images.shape[0] / batch_size)
    
    epoch = 0
    for i in range(num_epochs):
        for j in range(num_batch):
            batch_data = train_images[j*batch_size : (j+1)*batch_size]
            batch_label = train_labels[j*batch_size : (j+1)*batch_size]
            sess.run(train, feed_dict = {x: batch_data, y: batch_label, keep_prob: 0.5})
            if epoch % 100 == 0:
                print(sess.run(cross_entropy, feed_dict = {x: train_images, y: train_labels, keep_prob: 1.0}))
                train_accuracy = sess.run(accuracy, feed_dict = {x: train_images, y: train_labels, keep_prob: 1.0})
                print("Epoch %d, Training accuracy %g" % (epoch, train_accuracy))
            epoch += 1
        if num_batch * batch_size < train_images.shape[0]:
            batch_data = train_images[num_batch * batch_size : train_images.shape[0]]
            batch_label = train_labels[num_batch * batch_size : train_images.shape[0]]
            sess.run(train, feed_dict = {x: batch_data, y: batch_label, keep_prob: 0.5})
            if epoch % 100 == 0:
                print(sess.run(cross_entropy, feed_dict = {x: train_images, y: train_labels, keep_prob: 1.0}))
                train_accuracy = sess.run(accuracy, feed_dict = {x: train_images, y: train_labels, keep_prob: 1.0})
                print("Epoch %d, Training accuracy %g" % (epoch, train_accuracy))
            epoch += 1
    
    print("---Train end---")
    print('Testing accuracy %g' % sess.run(accuracy, feed_dict = {x: test_images, y: test_labels}))
    
    saver = tf.train.Saver()
    saver.save(sess, './model/My_Model')


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/My_Model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    
    graph = tf.get_default_graph()
    x_new = graph.get_tensor_by_name('x:0')
    y_new = graph.get_tensor_by_name('y_conv:0')
    keep_prob_new = graph.get_tensor_by_name('keep_prob:0')
    
    pre_y = sess.run(y_new, feed_dict = {x_new: test_images, keep_prob_new: 1.0})
    print(np.mean(np.argmax(pre_y, axis = 1) == test_label))
