########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
import sys
import data_processing as dp

class tiny_vgg:
    def __init__(self, weights=None, sess=None, lr=5e-5, epochs=50, batch=100, decay=0.7, keep_rate=0.15):
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.batch = batch
        self.keep_rate = keep_rate
        self.convlayers()
        self.fc_layers()
        self.data = dp.read_cifar10_data()
        self.probs = tf.nn.softmax(self.fc1l)
        self.sess = sess

        # if weights is not None and sess is not None:
        #     self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_drop_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        # # zero-mean input
        # # todo - this probably should be changed b/c it doesn't apply to our new dataset
        # with tf.name_scope('preprocess') as scope:
        #     resized = tf.image.resize_images(self.x, [224, 224])

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('dec1_1') as scope:
            conv = tf.nn.conv2d_transpose(self.conv1_1, kernel, [-1, 32, 32, 3], [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_1 = tf.nn.relu(out, name=scope)

            vars = tf.trainable_variables()
            whole = [v for v in vars if v.name.startswith("conv1_1") or v.name.startswith("dec1_1")]

            self.loss_1 = tf.reduce_mean(tf.squared_difference(self.dec_1, self.x))
            self.train_step1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_1, var_list=whole)

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('dec2_1') as scope:
            conv = tf.nn.conv2d_transpose(self.conv2_1, kernel, [-1, 16, 16, 64], [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_2 = tf.nn.relu(out, name=scope)

            vars = tf.trainable_variables()
            whole = [v for v in vars if v.name.startswith("conv2_1") or v.name.startswith("dec2_1")]

            self.loss_2 = tf.reduce_mean(tf.squared_difference(self.dec_2, self.pool1))
            self.train_step2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_2, var_list=whole)

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('dec3_1') as scope:
            conv = tf.nn.conv2d_transpose(self.conv3_1, kernel, [-1, 8, 8, 128], [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_3_1 = tf.nn.relu(out, name=scope)

            vars = tf.trainable_variables()
            whole = [v for v in vars if v.name.startswith("conv3_1") or v.name.startswith("dec3_1")]

            self.loss_3 = tf.reduce_mean(tf.squared_difference(self.dec_3_1, self.pool2))
            self.train_step3 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_3, var_list=whole)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('dec3_2') as scope:
            conv = tf.nn.conv2d_transpose(self.conv3_2, kernel, [-1, 8, 8, 256], [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_3_2 = tf.nn.relu(out, name=scope)

            vars = tf.trainable_variables()
            whole = [v for v in vars if v.name.startswith("conv3_2") or v.name.startswith("dec3_2")]

            self.loss_4 = tf.reduce_mean(tf.squared_difference(self.dec_3_2, self.conv3_1))
            self.train_step4 = tf.train.AdamOptimizer(self.lr).minimize(self.loss_4, var_list=whole)

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    def fc_layers(self):
        # fc1
        # the output of this layer must be changed in order to accommodate the fact that cifar-10 only has 10 labels
        with tf.name_scope('fc1') as scope:
            fc1w = tf.Variable(tf.truncated_normal([4096, 10], dtype=tf.float32, stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(tf.matmul(tf.layers.flatten(self.pool3), fc1w), fc1b)
            self.fc1l = tf.layers.batch_normalization(out, training=self.training)
            self.parameters += [fc1w, fc1b]

        vars = tf.trainable_variables()
        whole = [v for v in vars if v.name.startswith("fc1")]

        out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.fc1l))
        self.train_step5 = tf.train.AdamOptimizer(self.lr).minimize(out, var_list=whole)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(out)

    # def load_weights(self, weight_file, sess):
    #     weights = np.load(weight_file)
    #     keys = sorted(weights.keys())
    #     for i, k in enumerate(keys):
    #         print i, k, np.shape(weights[k])
    #         sess.run(self.parameters[i].assign(weights[k]))

    # code adapted from Liu Liu's LeNet implementation
    def train(self, report_freq=100):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        lr_l = []
        train_result = []
        valid_result = []

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # train layer 1
        sys.stdout.write('layer 1: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step1, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()

        # train layer 2
        sys.stdout.write('layer 2: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step2, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()

        # train layer 3
        sys.stdout.write('layer 3: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step3, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()

        # train layer 4
        sys.stdout.write('layer 4: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step4, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()

        # train classifier
        sys.stdout.write('classifier: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step5, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()

        # tune full network
        sys.stdout.write('full net: ')
        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch, labels = self.data.next_batch(self.batch)

            j = 0
            while batch != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                # self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: dp.randNoise(self.data.unitNormalize(batch), 0.05), self.y_: labels, self.training : True, self.keep_drop_prob : self.keep_rate})
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: self.data.unitNormalize(batch), self.y_: labels, self.training: True, self.keep_drop_prob: self.keep_rate})

                batch, labels = self.data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()
            if i % report_freq == 0:

                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.data.unitNormalize(self.data.train_X[:10000]), self.y_: self.data.train_y[:10000], self.training : False, self.keep_drop_prob : 1})
                train_result.append(train_acc)

                valid_acc = self.valid_eval()
                valid_result.append(valid_acc)

                lr_l.append(self.lr)

                print('step %d, training accuracy %g' % (i, train_acc))
                print('step %d, validation accuracy %g' % (i, valid_acc))
                print('step %d, learning rate %g' % (i, self.lr))

        return train_result, valid_result, lr_l

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()

        # for i in range(0,10000,50):
        # for i in range(0, 500, 50):
        ave = self.sess.run(self.accuracy, feed_dict={self.x: self.data.unitNormalize(self.data.test_X), self.y_: self.data.test_y, self.training : False, self.keep_drop_prob : 1})

        # ave = np.array(average).mean()

        print('Test accuracy accuracy %g' % (ave))

        return ave

    def use(self, input, target):
        self.eval()

        # for i in range(0,10000,50):
        # for i in range(0, 500, 50):
        ave = self.sess.run(self.accuracy, feed_dict={self.x: input, self.y_: target, self.training: False, self.keep_drop_prob: 1})

        # ave = np.array(average).mean()

        print('Test accuracy accuracy %g' % (ave))

        return ave

    def valid_eval(self):
        self.eval()

        # for i in range(0,10000,50):
        # for i in range(0, 500, 50):
        ave = self.sess.run(self.accuracy, feed_dict={self.x: self.data.unitNormalize(self.data.valid_X), self.y_: self.data.valid_y, self.training : False, self.keep_drop_prob : 1})
        # print(ave)

        # ave = np.array(average).mean()

        return ave

if __name__ == '__main__':
    print("initializing network")
    sess = tf.Session()
    tf.reset_default_graph()
    net = tiny_vgg(sess=sess)
    print("training")
    train_acc, valid_acc, lr_l = net.train(1)
    print("testing")
    test_acc = net.test_eval()

    # with open("noisy.csv", 'w') as o:
    with open("normal.csv", 'w') as o:
        buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
        o.write(buffer)

        buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
        o.write(buffer)

        buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
        o.write(buffer)

        buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
        o.write(buffer)

        buffer = str(test_acc)+'\n'
        o.write(buffer)
        o.flush()

    # sess = tf.Session()
    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    #
    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    #
    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]
