import tensorflow as tf

from data_processing import cifar_10_data
from task_1 import my_autoencoder

from PIL import Image

if __name__=="__main__":
    graph = tf.get_default_graph()
    sess = tf.Session()

    data = cifar_10_data()

    # first need initialize the autoencoder with saved weights
    net = my_autoencoder(sess=sess, task=2, restore=True, restore_path='.tmp/task_1/model_epoch_49.ckpt')

    original_image = Image.fromarray(data.train_X[0])
    original_image.show()
    # plt.imshow(original_image, interpolation='nearest')
    # plt.show()
    # a=raw_input("press enter: ")
    distorted_image = data.fetch_noisy_train_data(1)
    # plt.close()
    # plt.imshow(distorted_image)
    # plt.show()

