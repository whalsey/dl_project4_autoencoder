import numpy as np
import tensorflow as tf
import random
# from scipy.misc import imresize

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

# reading the data in in this form does requires too much memory (apparently)
# will read in data in natural form and will do the padding JIT for training/testing
def pad_image(dataset, padding):
    return np.pad(dataset, ((0,), (padding,), (padding,), (0,)), mode='constant', constant_values=0)
            
def resize_image(dataset, new_size):
    return tf.image.resize_images(dataset, new_size, method=tf.image.resize_bilinear)

def randShift(dataset):
    horlen = dataset.shape[1]
    verlen = dataset.shape[2]

    hor = random.randint(-3, 3)

    ver = random.randint(-3, 3)
    tmp = np.roll(dataset, shift=(hor), axis=(2))
    tmp = np.roll(tmp, shift=(ver), axis=(1))

    if hor > 0:
        for i in range(hor):
            tmp[:,:,i,:] = tmp[:,:,hor,:]
    else:
        for i in range(horlen+hor, horlen, 1):
            tmp[:,:,i,:] = tmp[:,:,hor-1,:]

    if ver > 0:
        for i in range(ver):
            tmp[:,i,:,:] = tmp[:,ver,:,:]
    else:
        for i in range(verlen+ver, verlen, 1):
            tmp[:,i,:,:] = tmp[:,ver-1,:,:]

    return tmp

def randNoise(dataset, stddev=0.01):
    return dataset + np.random.normal(0, stddev, dataset.shape)

class cifar_10_data:
    def __init__(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0

        sess = tf.Session()

        with tf.device('/cpu:0'):
            data = unpickle("cifar-10-batches-py/data_batch_1")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.valid_X = tmp
            self.valid_y = tf.one_hot(data['labels'], 10).eval(session=sess)

            data = unpickle("cifar-10-batches-py/data_batch_2")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.train_X = tmp
            self.train_y = tf.one_hot(data['labels'], 10).eval(session=sess)

            data = unpickle("cifar-10-batches-py/data_batch_3")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.train_X = np.concatenate((self.train_X, tmp))
            self.train_y = np.concatenate((self.train_y, tf.one_hot(data['labels'], 10).eval(session=sess)))

            data = unpickle("cifar-10-batches-py/data_batch_4")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.train_X = np.concatenate((self.train_X, tmp))
            self.train_y = np.concatenate((self.train_y, tf.one_hot(data['labels'], 10).eval(session=sess)))

            data = unpickle("cifar-10-batches-py/data_batch_5")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.train_X = np.concatenate((self.train_X, tmp))
            self.train_y = np.concatenate((self.train_y, tf.one_hot(data['labels'], 10).eval(session=sess)))

            data = unpickle("cifar-10-batches-py/test_batch")
            tmp = data['data'].reshape((-1, 32, 32, 3))
            self.test_X = tmp
            self.test_y = tf.one_hot(data['labels'], 10).eval(session=sess)

        self._num_examples = self.train_X.shape[0]

        return

    # code addapted from https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data#43538941
    def next_batch(self, batch_size):
        start = self._index_in_epoch

        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            self._data = self.train_X[idx]  # get list of `num` random samples
            self._labels = self.train_X[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.train_X[idx0]  # get list of `num` random samples
            self._labels = self.train_X[idx0]

            self._index_in_epoch = 0  # avoid the case where the #sample != integer times of batch_size

            return [], []
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # add small perturbations to the data before passing out
            # as a method of "augmenting" the dataset but without
            # actually changing the training set size
            return self._data[start:end], self._labels[start:end]

    def next_noisyBatch(self, batchsize, stddev=0.01):
        input, targets = self.next_batch(batchsize)
        return randNoise(input, stddev=stddev), targets

    def next_droppedBatch(self, batchsize, drop=0.01):
        pass

    def get_mean(self):
        return self.train_X.mean(axis=(0, 1, 2))

    def get_stddev(self):
        return self.train_X.std(axis=(0, 1, 2))

    def normalize(self):
        mean = self.get_mean()
        std = self.get_stddev()
        self.train_X = np.subtract(self.train_X, mean)
        self.valid_X = np.subtract(self.valid_X, mean)
        self.test_X = np.subtract(self.test_X, mean)

        self.train_X = self.train_X / std
        self.valid_X = self.valid_X / std
        self.test_X = self.test_X / std

def read_cifar10_data():
    return cifar_10_data()

if __name__=="__main__":
    # test = read_cifar10_data()
    #
    # hello = test.get_mean()
    # for _ in range(5):
    #     input, labels = test.next_batch(20000)
    #
    #     while input != []:
    #         print(input)
    #         print(input.shape)
    #
    #         input, labels = test.next_batch(20000)
    #
    #     print("done")

    test = np.arange(150).reshape((3,5,5,2))

    output = randShift(test)
