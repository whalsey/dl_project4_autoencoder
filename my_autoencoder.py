import tensorflow as tf

class my_autoencoder:

    def __init__(self, sess=None):


    def initNetStructure(self):
        self.input = tf.placeholder(tf.float32, (32,32,3), 'input')



    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    print('HELLO PEA!!')
