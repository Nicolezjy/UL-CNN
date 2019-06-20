from skimage import io,transform
import os
import glob
import numpy as np
import tensorflow as tf
import datetime
import cv2
import warnings
warnings.filterwarnings("ignore")


now = datetime.datetime.now()

w = 120
h = 120
c = 3

train_path = "D:/Project/CNN/Train/"
test_path = "D:/Project/CNN/Test/"
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.jpg'):
            #print("reading the image:%s" % img)
            image = io.imread(img)
            image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)

train_data,train_label = read_image(train_path)
test_data,test_label = read_image(test_path)
print(train_label)

train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

x = tf.placeholder(tf.float32,[None,w,h,c],name='x')
y_ = tf.placeholder(tf.int32,[None],name='y_')


def inference(input_tensor,train,regularizer):


    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,3,8],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[8],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable('weight',[7,7,8,8],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[8],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('layer3-pool1'):
        pool1 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer4-conv3'):
        conv3_weights = tf.get_variable('weight',[9,9,8,3],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias',[3],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1,conv3_weights,strides=[1,1,1,1],padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))


    with tf.name_scope('layer5-pool2'):
        pool2 = tf.nn.max_pool(relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


   
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])

    
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[512],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    
    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable('weight',[512,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2,0.5)

    
    with tf.variable_scope('layer8-fc3'):
        fc3_weights = tf.get_variable('weight',[32,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[2],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases
    return logit


regularizer = tf.contrib.layers.l2_regularizer(0.001)

y = inference(x,False,regularizer)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

saver = tf.train.Saver(max_to_keep=1)

sess = tf.InteractiveSession()
 
sess.run(tf.global_variables_initializer())

   
train_num = 200
batch_size = 128

def train():
    for i in range(train_num):
        print(i)
        train_loss, train_acc, batch_num = 0, 0, 0
        for train_data_batch, train_label_batch in get_batch(train_data, train_label, batch_size):
            _, err, acc = sess.run([train_op, loss, accuracy], feed_dict={x: train_data_batch, y_: train_label_batch})
            train_loss += err;
            train_acc += acc;
            batch_num += 1
        print("train loss:", train_loss / batch_num)
        print("train acc:", train_acc / batch_num)
        saver.save(sess, "model/model1")


def test():
    test_num = 1
    batch_size1 = 400
    saver.restore(sess, "model/model1")

    for i in range(test_num):

        test_loss, test_acc, batch_num = 0, 0, 0
        for test_data_batch, test_label_batch in get_batch(test_data, test_label, batch_size1):
            err, acc = sess.run([loss, accuracy], feed_dict={x: test_data_batch, y_: test_label_batch})
            print(acc)
            test_loss += err;
            test_acc += acc;

        print("test loss:", test_loss / 1)
        print("test acc:", test_acc / 1)

if __name__ == "__main__":
    start = datetime.datetime.now()
    test()
    end = datetime.datetime.now()
    print(end - start)
    print((end - start).seconds)
    print((end - start).microseconds)




