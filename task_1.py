import tensorflow as tf
import sys
import os
from data_processing import cifar_10_data

import logging
logging.basicConfig(level=logging.DEBUG)

cur_dir = os.path.curdir

class my_autoencoder:

    def __init__(self, sess=None, latent_dim=1024, learning_rate=1e-4, epochs=50, batch_size=100, data=None, task=None, restore=False, restore_path=None):
        self.sess = sess
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data

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

        with tf.name_scope('Encoder_1'):
            flat = tf.reshape(self.x, [-1, 3072])
            self.weights_1 = tf.Variable(tf.truncated_normal([3072, 2048]), name='weights_1')
            self.enc1_bias = tf.Variable(tf.truncated_normal([2048]), name='enc1_bias')

            self.encoded_1 = tf.nn.sigmoid(tf.add(tf.matmul(flat, self.weights_1), self.enc1_bias), name='encoded_1')

        with tf.name_scope('Encoder_2'):
            self.weights_2 = tf.Variable(tf.truncated_normal([2048, self.latent_dim]), name='weights_2')
            self.enc2_bias = tf.Variable(tf.truncated_normal([self.latent_dim]), name='enc2_bias')

            encoded = tf.nn.sigmoid(tf.add(tf.matmul(self.encoded_1, self.weights_2), self.enc2_bias), name='encoded_2')

        with tf.name_scope('Decoder_2'):
            self.dec2_bias = tf.Variable(tf.truncated_normal([2048]), name='dec3_bias')
            self.decoded_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoded, tf.transpose(self.weights_2)), self.dec2_bias), name='decoded_2')

        with tf.name_scope('Decoder_1'):
            self.dec1_bias = tf.Variable(tf.truncated_normal([3072]), name='dec1_bias')
            decoded = tf.nn.sigmoid(tf.add(tf.matmul(self.decoded_2, tf.transpose(self.weights_1)), self.dec1_bias), name='decoded_1')

        self.y = tf.reshape(decoded, [-1, 32, 32, 3], name='output')

        self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.y_, self.y), 2))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):

        train_acc_record = []
        valid_acc_record = []

        for epoch in range(self.epochs):
            sys.stdout.write("Epoch: {} ".format(epoch))
            sys.stdout.flush()

            input_im, output_im = self.data.next_noisyBatch(self.batch_size, stddev=0.05)

            i = 0
            while input_im != [] and output_im != []:

                self.sess.run(self.train_step, feed_dict={self.x: input_im, self.y_: output_im})

                if i % 100 == 0:
                    sys.stdout.write('=')
                    sys.stdout.flush()

                i += 1
                input_im, output_im = self.data.next_noisyBatch(self.batch_size, stddev=0.05)

            # calculate training loss
            X, y_ = self.data.fetch_noisy_train_data(10000)
            train_acc = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})

            # calculate validation loss
            X, y_ = self.data.fetch_noisy_valid_data()
            valid_acc = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})

            train_acc_record.append(train_acc)
            valid_acc_record.append(valid_acc)

            sys.stdout.write('\nTraining loss: {}, Validation loss: {}\n'.format(train_acc, valid_acc))

            save_path = "./tmp/task_{}/model_epoch_{}.ckpt".format(self.task, epoch)
            if not os.path.exists('/'.join(save_path.split('/')[:-1])):
                os.makedirs('/'.join(save_path.split('/')[:-1]))
            self.saver.save(sess, save_path=save_path)

        return train_acc_record, valid_acc_record

    def test(self):
        X, y_ = self.data.fetch_noisy_test_data()
        test_loss = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})
        sys.stdout.write('Test loss: {}\n'.format(test_loss))
        return test_loss

    def use(self, input):
        output = self.sess.run(self.y, feed_dict={self.x: input})

        return output

if __name__ == "__main__":
    # TASK 1
    # read in the data - may try several different types and ranges for distortion
    data = cifar_10_data()
    data.unitNormalize()

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




