import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import sys
import os
from data_processing import cifar_10_data
import data_processing as dp

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)

cur_dir = os.path.curdir

class my_autoencoder:

    def __init__(self, sess=None, latent_dim=512, learning_rate=1e-4, epochs=150, batch_size=100, stddev=0.05, data=None, task=None, restore=False, restore_path=None):
        self.sess = sess
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.stddev = stddev

        if task is None:
            logging.error("No task enumerated!")
            exit(-1)
        else:
            self.task = task

        self.initNetStructure()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.saver = tf.train.Saver()

        if restore and restore_path is not None:
            self.saver.restore(sess=self.sess, save_path=restore_path)

    def initNetStructure(self):
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_layer')
        self.y_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='ground_truth')

        input = tf.layers.flatten(self.x)
        output = tf.layers.flatten(self.y_)

        with tf.name_scope('Encoder_1'):
            self.weights_1 = tf.Variable(tf.truncated_normal([3072, 2048], stddev=self.stddev), name='weights_1')
            self.enc1_bias = tf.Variable(tf.truncated_normal([2048], stddev=self.stddev), name='enc1_bias')

            self.encoded_1 = tf.nn.relu(tf.add(tf.matmul(input, self.weights_1), self.enc1_bias), name='encoded_1')

        with tf.name_scope('Encoder_2'):
            self.weights_2 = tf.Variable(tf.truncated_normal([2048, 1024], stddev=self.stddev), name='weights_2')
            self.enc2_bias = tf.Variable(tf.truncated_normal([1024], stddev=self.stddev), name='enc2_bias')

            self.encoded_2 = tf.nn.relu(tf.add(tf.matmul(self.encoded_1, self.weights_2), self.enc2_bias), name='encoded_2')

        with tf.name_scope('Encoder_3'):
            self.weights_3 = tf.Variable(tf.truncated_normal([1024, self.latent_dim], stddev=self.stddev), name='weights_3')
            self.enc3_bias = tf.Variable(tf.truncated_normal([self.latent_dim], stddev=self.stddev), name='enc3_bias')

            encoded = tf.nn.relu(tf.add(tf.matmul(self.encoded_2, self.weights_3), self.enc3_bias), name='encoded_3')

        with tf.name_scope('Decoder_3'):
            self.dec3_bias = tf.Variable(tf.truncated_normal([1024], stddev=self.stddev), name='dec3_bias')
            self.decoded_3 = tf.nn.relu(tf.add(tf.matmul(encoded, tf.transpose(self.weights_3)), self.dec3_bias), name='decoded_2')

        with tf.name_scope('Decoder_2'):
            self.dec2_bias = tf.Variable(tf.truncated_normal([2048], stddev=self.stddev), name='dec3_bias')
            self.decoded_2 = tf.nn.relu(tf.add(tf.matmul(self.decoded_3, tf.transpose(self.weights_2)), self.dec2_bias), name='decoded_2')

        with tf.name_scope('Decoder_1'):
            self.dec1_bias = tf.Variable(tf.truncated_normal([3072], stddev=self.stddev), name='dec1_bias')
            self.decoded = tf.nn.relu(tf.add(tf.matmul(self.decoded_2, tf.transpose(self.weights_1)), self.dec1_bias), name='decoded_1')

        self.y = tf.reshape(self.decoded, [-1, 32, 32, 3], name='output')

        self.loss = tf.reduce_mean(tf.squared_difference(output, self.decoded))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):

        train_acc_record = []
        valid_acc_record = []

        for epoch in range(self.epochs):
            sys.stdout.write("Epoch: {} ".format(epoch))
            sys.stdout.flush()

            raw_im, _ = self.data.next_batch(self.batch_size)

            i = 0
            while raw_im != []:
                output_im = self.data.unitNormalize(raw_im)
                input_im = dp.randNoise(output_im, stddev=self.stddev)

                self.sess.run(self.train_step, feed_dict={self.x: input_im, self.y_: output_im})
                # output = self.sess.run(self.decoded, feed_dict={self.x: input_im, self.y_: output_im})
                # loss = self.sess.run(self.loss, feed_dict={self.x: input_im, self.y_: output_im})

                if i % 100 == 0:
                    sys.stdout.write('=')
                    sys.stdout.flush()

                i += 1
                raw_im, _ = self.data.next_batch(self.batch_size)

            # calculate training loss
            y_ = self.data.unitNormalize(self.data.train_X[:10000])
            X = dp.randNoise(y_, stddev=self.stddev)
            train_acc = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})

            # calculate validation loss
            y_ = self.data.unitNormalize(self.data.valid_X)
            X = dp.randNoise(y_, stddev=self.stddev)
            valid_acc = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})

            train_acc_record.append(train_acc)
            valid_acc_record.append(valid_acc)

            sys.stdout.write('\nTraining loss: {}, Validation loss: {}\n'.format(train_acc, valid_acc))

            save_path = "./tmp/task_{}/model_epoch_{}.ckpt".format(self.task, epoch)
            if not os.path.exists('/'.join(save_path.split('/')[:-1])):
                os.makedirs('/'.join(save_path.split('/')[:-1]))
            self.saver.save(sess, save_path=save_path)

            choice = np.random.choice(10000, 10, replace=False)
            # choice = np.random.choice(10, 10, replace=False)
            save_img = "./img/img_{}/epoch_{}.png".format(task, epoch)
            if not os.path.exists('/'.join(save_img.split('/')[:-1])):
                os.makedirs('/'.join(save_img.split('/')[:-1]))
            self.visualize(self.data.test_X[choice], save_img)

        return train_acc_record, valid_acc_record

    def test(self):
        y_ = self.data.unitNormalize(self.data.test_X)
        X = dp.randNoise(y_, stddev=self.stddev)
        test_loss = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})
        sys.stdout.write('Test loss: {}\n'.format(test_loss))
        return test_loss

    def use(self, input):
        output = self.sess.run(self.y, feed_dict={self.x: input})

        return output

    def visualize(self, input, path=None, show=False):

        # plot originals
        samples = input.shape[0]

        fig, ax = plt.subplots(3, samples)

        for i in range(samples):
            ax[0, i].imshow(input[i])

        # distort originals
        distorted = dp.randNoise(self.data.unitNormalize(input), self.stddev)
        distorted_im = self.data.unitUnnormalize(distorted).astype(np.uint8)

        for i in range(samples):
            ax[1, i].imshow(distorted_im[i])

        # get output
        output = self.use(distorted)
        output_im = self.data.unitUnnormalize(output).astype(np.uint8)

        for i in range(samples):
            ax[2, i].imshow(output_im[i])

        if path is not None:
           fig.savefig(path, dpi=(fig.dpi)*2)

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # TASK 1
    # read in the data - may try several different types and ranges for distortion
    data = cifar_10_data()

    graph = tf.get_default_graph()
    sess = tf.Session()

    task = 1
    net = my_autoencoder(sess=sess, data=data, task=task)

    train_loss, valid_loss = net.train()
    test_loss = net.test()

    sess.close()

    # write out results

    file = "./output/task_{}_output.csv".format(1)
    if not os.path.exists('/'.join(file.split('/')[:-1])):
        os.makedirs('/'.join(file.split('/')[:-1]))

    def write(file, train, valid, test):
        epochs = [i for i in range(len(train))]

        with open(file, 'w') as o:
            buffer = ','.join(["epochs"] + [i.__str__() for i in epochs]) + '\n'
            o.write(buffer)
            o.flush()

            buffer = ','.join(["train loss"] + [i.__str__() for i in train]) + '\n'
            o.write(buffer)
            o.flush()

            buffer = ','.join(["valid loss"] + [i.__str__() for i in valid]) + '\n'
            o.write(buffer)
            o.flush()

            buffer = ','.join(["test loss"] + [test.__str__()]) + '\n'
            o.write(buffer)
            o.flush()

    write(file, train_loss, valid_loss, test_loss)


    # TASK 2




