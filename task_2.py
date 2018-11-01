import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np

from tiny_vgg import tiny_vgg
from task_1 import my_autoencoder
from data_processing import cifar_10_data
import data_processing as dp

if __name__ == '__main__':
    # initialize and load ae and filter test set
    sess = tf.Session()

    data = cifar_10_data()

    # first need initialize the autoencoder with saved weights
    net2 = my_autoencoder(sess=sess, task=2, restore=True, data=data, restore_path='./tmp/task_1/model_epoch_149.ckpt')

    # train vgg
    print("initializing network")
    sess = tf.Session()
    tf.reset_default_graph()
    net = tiny_vgg(sess=sess)
    print("training")
    train_acc, valid_acc, lr_l = net.train(1)
    print("testing")
    test_acc = net.test_eval()

    with open("scratchnet2.csv", 'w') as o:
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

    tf.reset_default_graph()

    # calculate accuracy of denoised images
    net2.visualize(data.test_X[:10], './img/img_2/img_2')
    new_images = net2.use(data.unitNormalize(data.test_X))
    target = data.test_y

    tf.reset_default_graph()

    results = net.use(new_images, target)
    with open("something.csv", 'w') as o:
        buffer = results.__str__()
        o.write(buffer)
        o.flush()