import tensorflow as tf
import numpy as np

from data_processing import cifar_10_data
from task_1 import my_autoencoder

from PIL import Image

if __name__=="__main__":
    graph = tf.get_default_graph()
    sess = tf.Session()

    data = cifar_10_data()

    # first need initialize the autoencoder with saved weights
    net = my_autoencoder(sess=sess, task=2, restore=True, restore_path='./tmp/task_1/model_epoch_49.ckpt')

    tmp1 = data.train_X[0]
    original_image = Image.fromarray(tmp1)
    original_image.show()
    raw_input("press enter")
    # plt.imshow(original_image, interpolation='nearest')
    # plt.show()
    # a=raw_input("press enter: ")
    data.unitNormalize()
    blah = data.fetch_noisy_train_data(1)
    tmp2, clean = blah[0], blah[1]
    clean = data.unitUnnormalize(clean)
    clean_image = Image.fromarray(clean[0].astype(np.uint8))
    clean_image.show()
    raw_input("press enter")
    tmp = data.unitUnnormalize(tmp2[0]).astype(np.uint8)
    distorted_image = Image.fromarray(tmp)
    distorted_image.show()
    raw_input("press enter")
    new = data.unitUnnormalize(net.use(tmp2)).astype(np.uint8)
    new_image = Image.fromarray(new[0])
    new_image.show()
    # plt.close()
    # plt.imshow(distorted_image)
    # plt.show()
    raw_input("press enter")

