import tensorflow as tf
class MNISTDataset(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = tf.reshape(x_train, (x_train.shape[0], 784))
        self.x_test = tf.reshape(x_test, (x_test.shape[0], 784))

    def get_train_set(self, n_batchs=32):
        return  tf.data.Dataset.from_tensor_slices(
        (self.x_train, self.y_train)).shuffle(10000).batch(n_batchs)
    
    def get_test_set(self, n_batchs=32):
        return  tf.data.Dataset.from_tensor_slices(
        (self.x_test, self.y_test)).shuffle(10000).batch(n_batchs)
    