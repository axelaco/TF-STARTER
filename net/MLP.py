import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, num_class):
        super(MLP, self).__init__(name='MLP')
        self.fc1 = tf.keras.layers.Dense(128, input_shape=(784,))
        self.fc2 = tf.keras.layers.Dense(256)
        if num_class > 1:
            self.out = tf.keras.layers.Dense(num_class, activation='softmax')
        else:
            self.out = tf.keras.layers.Dense(num_class, activation='sigmoid')
    
    def call(self, input, training):
        x = self.fc1(input)
        x = self.fc2(x)
        return self.out(x)