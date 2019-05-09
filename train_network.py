import numpy as np
import tensorflow as tf
import time as time
from datetime import datetime

from data_wrapper import DataReader

reader = DataReader()
graph = tf.Graph()
session = tf.Session(graph=graph)
start_time = datetime.now().strftime('%m-%d-%H-%M-%S')

# global parameters
batch_size = 65
learning_rate = .0005
num_training_steps = 5 * 10**5

# first conv-pool architectural hyper-parameters
conv1_size = [8, 8, 1, 16]
pool1_size = [1, 8, 8, 1]
pool1_stride = [1, 4, 4, 1]

# second conv-pool architectural hyper=parameters
conv2_size = [5, 5, 16, 64]
pool2_size = [1, 4, 4, 1]
pool2_stride = [1, 2, 2, 1]
pool2_flat = [-1, 8**2 * 64]

# fully connected layer architectural hyper-parameters
fc_hidden_units = 48
output_units = 2


def fully_connected_layer(inputs, shape, layer_name):
    W = tf.get_variable(
        name='{}_weights'.format(layer_name),
        shape=shape,
        initializer=tf.random_normal_initializer(
            stddev=1.0 / np.sqrt(shape[0])
        )
    )
    b = tf.get_variable(
        name='{}_biases'.format(layer_name),
        shape=shape[-1],
        initializer=tf.constant_initializer(value=0.1)
    )
    z = tf.add(tf.matmul(inputs, W), b, name='{}_z'.format(layer_name))
    a = tf.nn.relu(z)
    return z, a


def convolutional_layer(inputs, shape, layer_name):
    W = tf.get_variable(
        name='{}_weights'.format(layer_name),
        shape=shape,
        initializer=tf.truncated_normal_initializer(
            stddev=0.1
        )
    )
    b = tf.get_variable(
        name='{}_biases'.format(layer_name),
        shape=shape[-1],
        initializer=tf.constant_initializer(value=0.1)
    )
    z = tf.nn.conv2d(inputs, W, strides=[1, 2, 2, 1], padding='SAME') + b
    a = tf.nn.relu(z)
    return a


def pooling_layer(inputs, shape, stride):
    return tf.nn.max_pool(inputs, ksize=shape, strides=stride, padding='SAME')


def feedforward(inputs, keep_prob, features):
    a_conv1 = convolutional_layer(inputs, conv1_size, layer_name='conv1')
    a_pool1 = pooling_layer(a_conv1, pool1_size, pool1_stride)
    a_conv2 = convolutional_layer(a_pool1, conv2_size, layer_name='conv2')
    a_pool2 = pooling_layer(a_conv2, pool2_size, pool2_stride)
    a_pool2_ = tf.reshape(a_pool2, pool2_flat)

    a_pool_2_features = tf.concat(1, [a_pool2_, tf.reshape(features, [-1, 4])])
    _, a_fc = fully_connected_layer(
        inputs=a_pool_2_features,
        shape=[pool2_flat[1] + 4, fc_hidden_units],
        layer_name='fc1'
    )

    a_fc_drop = tf.nn.dropout(a_fc, keep_prob)
    y_, _ = fully_connected_layer(
        inputs=a_fc_drop,
        shape=[fc_hidden_units, output_units],
        layer_name='fc2'
    )
    return y_


def backprop(y_, y):
    cross_ent = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y_, y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_ent)
    return train_step


def evaluate(preds, truth):
    correct_prediction = tf.equal(tf.argmax(preds, 1), truth)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


with graph.as_default():

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('conv_net'):
        labels, images, features = reader.get_train_batch(batch_size)
        y_ = feedforward(images, keep_prob, features)
        train_step = backprop(y_, labels)
        preds = tf.argmax(y_, 1)
        accuracy = evaluate(y_, labels)

    with tf.variable_scope('conv_net', reuse=True):
        test_labels, test_images, test_features = \
            reader.get_test_batch(batch_size)
        test_y_ = feedforward(test_images, keep_prob, test_features)
        test_preds = tf.argmax(test_y_, 1)
        test_accuracy = evaluate(test_y_, test_labels)

    # pipeline for feeding images for future prediction
    with tf.variable_scope('conv_net', reuse=True):
        feed_images = tf.placeholder(
            tf.float32, shape=[None, 64, 64, 1], name='images')
        feed_features = tf.placeholder(
            tf.float32, shape=[None, 4], name='features')
        feed_logits = feedforward(feed_images, keep_prob, feed_features)
        feed_logits = tf.identity(feed_logits, name='logits')

    saver = tf.train.Saver()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())


with session.as_default():

    session.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    cummulative_accuracy = 0
    last_time = time.time()

    warm_start_init_step = 0
    if warm_start_init_step != 0:
        ckpt_file = 'checkpoints/model-{}'.format(warm_start_init_step)
        saver.restore(session, ckpt_file)

    for step_num in range(num_training_steps):

        _, acc = session.run(
            [train_step, accuracy],
            feed_dict={keep_prob: 1.0}
        )
        cummulative_accuracy += acc

        # log to stdout for sanity checks every 500 steps
        log_interval = 1
        if step_num % log_interval == 0:
            print (step_num,)
            print (cummulative_accuracy / float(log_interval),)
            print ('({} s)'.format(time.time() - last_time))
            last_time = time.time()
            cummulative_accuracy = 0

        # train and test evaluation on sampled data every 1000 steps
        eval_interval = 1000
        eval_sample_size = 250
        if step_num % eval_interval == 0:
            train_acc = 0
            num_train = 3493 * 60
            train_iters = min(eval_sample_size, num_train / batch_size)
            for i in range(train_iters):
                train_acc += session.run(
                    [accuracy],
                    feed_dict={keep_prob: 1.0}
                )[0]
            train = train_acc / train_iters

            test_acc = 0
            num_test = 350 * 60
            test_iters = min(eval_sample_size, num_test / batch_size)
            for i in range(test_iters):
                test_acc += session.run(
                    [test_accuracy],
                    feed_dict={keep_prob: 1.0}
                )[0]
            test = test_acc / test_iters

            with open('logs/train_log-{}.txt'.format(start_time), 'a') as log:
                log.write('{} {}\n'.format(step_num, train))
            with open('logs/test_log-{}.txt'.format(start_time), 'a') as log:
                log.write('{} {}\n'.format(step_num, test))

        # save model parameters every 10000 steps
        if step_num % 10 == 0:
            print ('asdf')
            saver.save(session, 'checkpoints/model', global_step=step_num)

    coord.request_stop()
    coord.join(threads)
