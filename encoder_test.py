import tensorflow as tf
import numpy as np

from data_processing import cifar_10_data
import data_processing as dp
from task_1 import my_autoencoder

from PIL import Image
import matplotlib.pyplot as plt

if __name__=="__main__":
    graph = tf.get_default_graph()
    sess = tf.Session()

    data = cifar_10_data()

    # first need initialize the autoencoder with saved weights
    net = my_autoencoder(sess=sess, task=2, restore=True, data=data, restore_path='./tmp/task_1/model_epoch_49.ckpt')

    training = data.train_X[:10]
    net.visualize(training, 'temp.png')

    exit(0)
    training = data.train_X[0]
    plt.imshow(training)
    plt.show()
    # training_image = Image.fromarray(training)
    # training_image.show()
    raw_input("press enter")

    restored = data.unitUnnormalize(data.unitNormalize(training))
    restored_image = Image.fromarray(restored.astype(np.uint8))
    restored_image.show()
    raw_input("press enter")

    distort = data.unitUnnormalize(dp.randNoise(data.unitNormalize(training))).astype(np.uint8)
    distort_image = Image.fromarray(distort)
    distort_image.show()
    raw_input("press enter")

    new = data.unitUnnormalize(net.use([dp.randNoise(data.unitNormalize(training))])).astype(np.uint8)
    new_image = Image.fromarray(new[0])
    new_image.show()
    raw_input("press enter")

    another = data.unitUnnormalize(net.use([data.unitNormalize(training)])).astype(np.uint8)
    another_image = Image.fromarray(new[0])
    another_image.show()
    raw_input("press enter")

