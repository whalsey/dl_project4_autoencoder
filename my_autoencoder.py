import tensorflow as tf
from data_processing import cifar_10_data

class my_autoencoder:

    def __init__(self, sess=None, latent_dim=2048, learning_rate=1e-4, epochs=50, batch_size=100, data=None):
        self.sess = sess
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.initNetStructure()

    def initNetStructure(self):
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_layer')
        self.y_ = tf.placeholder(tf.float32, [None, 32, 32, 3], name='ground_truth')

        with tf.name_scope('Encoder'):
            flat = tf.reshape(self.x, [-1, 3072])
            self.weights = tf.Variable(tf.truncated_normal_initializer([3027, self.latent_dim]), name='weights')
            self.bias = tf.Variable(tf.truncated_normal_initializer([self.latent_dim]), name='bias')

            encoded = tf.nn.sigmoid(tf.add(tf.matmul(flat, self.weights), self.bias), name='encoded')

        with tf.name_scope('Decoder'):
            decoded = tf.nn.sigmoid(tf.add(tf.matmul(encoded, tf.transpose(self.weights)), tf.transpose(self.bias)), name='decoded')

            self.y = tf.reshape(decoded, [-1, 32, 32, 3], name='output')

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y))
            self.cost = tf.

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):

        for epoch in range(self.epochs):
            input_im, output_im = self.data.next_noisyBatch(stddev=0.01)

            if input_im != [] and output_im != []:

                self.sess.run(self.train_step, feed_dict={self.x: input_im, self.y_: output_im})

    def validate(self):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    # read in the data - may try several different types and ranges for distortion
    data = cifar_10_data(stddev_noise=0.01)

    graph = tf.get_default_graph()
    # init_op = tf.global_variables_initializer()
    sess = tf.Session()
    # sess.run(init_op)

    net = my_autoencoder(sess=sess, data=data)

    for op in graph.get_operations():
        print(op.name)

    sess.close()
