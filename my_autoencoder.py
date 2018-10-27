import tensorflow as tf
import sys
import os
from data_processing import cifar_10_data

import logging
logging.basicConfig(level=logging.DEBUG)

cur_dir = os.path.curdir

class my_autoencoder:

    def __init__(self, sess=None, latent_dim=2048, learning_rate=1e-4, epochs=50, batch_size=100, data=None, task=None, restore=False):
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

        if restore:
            # todo - restore files
            pass

    def initNetStructure(self):
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_layer')
        self.y_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='ground_truth')

        with tf.name_scope('Encoder'):
            flat = tf.reshape(self.x, [-1, 3072])
            self.weights = tf.Variable(tf.truncated_normal([3072, self.latent_dim]), name='weights')
            self.enc_bias = tf.Variable(tf.truncated_normal([self.latent_dim]), name='end_bias')

            encoded = tf.nn.sigmoid(tf.add(tf.matmul(flat, self.weights), self.enc_bias), name='encoded')

        with tf.name_scope('Decoder'):
            self.dec_bias = tf.Variable(tf.truncated_normal([3072]), name='dec_bias')
            decoded = tf.nn.sigmoid(tf.add(tf.matmul(encoded, tf.transpose(self.weights)), self.dec_bias), name='decoded')

            self.y = tf.reshape(decoded, [-1, 32, 32, 3], name='output')

            self.loss = tf.reduce_mean(tf.pow(tf.subtract(self.y_, self.y), 2))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):

        train_acc_record = []
        valid_acc_record = []

        for epoch in range(self.epochs):
            sys.stdout.write("Epoch: {} ".format(epoch))
            sys.stdout.flush()

            input_im, output_im = self.data.next_noisyBatch(self.batch_size, stddev=0.1)

            i = 0
            while input_im != [] and output_im != []:

                self.sess.run(self.train_step, feed_dict={self.x: input_im, self.y_: output_im})

                if i % 100 == 0:
                    sys.stdout.write('=')
                    sys.stdout.flush()

                i += 1
                input_im, output_im = self.data.next_noisyBatch(self.batch_size, stddev=0.1)

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
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            self.saver.save(sess, save_path=save_path)

        return train_acc_record, valid_acc_record

    def test(self):
        X, y_ = self.data.fetch_noisy_test_data()
        test_loss = self.sess.run(self.loss, feed_dict={self.x: X, self.y_: y_})
        sys.stdout.write('Test loss: {}\n'.format(test_loss))
        return test_loss

if __name__ == "__main__":
    # TASK 1
    # read in the data - may try several different types and ranges for distortion
    data = cifar_10_data(stddev_noise=0.01)
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
    os.makedirs('/'.join(file.split('/')[:-1]), exist_ok=True)

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




