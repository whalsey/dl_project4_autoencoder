import tensorflow as tf

from data_processing import cifar_10_data
from task_1 import my_autoencoder

if __name__=="__main__":
    graph = tf.get_default_graph()
    sess = tf.Session()

    data = cifar_10_data()

    # first need initialize the autoencoder with saved weights
    net = my_autoencoder(sess=sess, task=2, restore=True, restore_path='.tmp/task_1/model_epoch_49.ckpt')

    data