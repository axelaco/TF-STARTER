import tensorflow as tf
from trainer.Trainer import Trainer
from net.MLP import MLP
from dataset.MNISTDataset import MNISTDataset
class Constructor(object):
    def __init__(self, config):
        self.lr = config.lr
        self.momentum = getattr(config, 'momentum', 0.0)
        self.trainer_id = config.trainer_id
        self.optimizer_id = config.optimizer_id
        self.network_id = config.network_id
        self.loss_id = config.loss_id
        self.num_classes = config.num_classes
        self.train_loader_id = config.train_loader_id
        self.test_loader_id = config.test_loader_id
        self.num_batchs = getattr(config, 'num_batchs', 32)

    def get_train_loader(self):
        if self.train_loader_id == 0:
            return MNISTDataset().get_train_set(self.num_batchs)
        else:
            raise NameError('Train Loader id unknown {}'.format(self.train_loader_id))


    def get_test_loader(self):
        if self.test_loader_id == 0:
            return MNISTDataset().get_test_set(self.num_batchs)
        else:
            raise NameError('Test Loader id unknown {}'.format(self.test_loader_id))
    
    def get_network(self):
        if self.network_id == 0:
            return MLP(self.num_classes)
        else:
            raise NameError('Network id unknown {}'.format(self.network_id))
    
    def get_optimizer(self):
        if self.optimizer_id == 0:
            # SGD Optimizer
            return tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)
        elif self.optimizer_id == 1:
            # RMS Prop Optimzer
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr, momentum=self.momentum)
        elif self.optimizer_id == 2:
            # Adam Optimizer
            return tf.keras.optimizers.Adam(self.lr)
        else:
            raise NameError('Optimizer id unknown {}'.format(self.optimizer_id))
    
    def get_trainer(self, config, model, optimizer, loss, train_loader, test_loader):
        if self.trainer_id == 0:
            return Trainer(config, model, optimizer, loss, train_loader, test_loader)
        else:
            raise NameError('Trainer id unknown {}'.format(self.trainer_id))
        
    def get_loss(self):
        if self.loss_id == 0:
            # Binary Cross Entropy
            return tf.keras.losses.BinaryCrossentropy()
        elif self.loss_id == 1:
            # Categorical Cross Entropy 
            # One hot encoded representation for labels
            # [[0, 0, 1], [1, 0, 0]]
            return tf.keras.losses.CategoricalCrossentropy()
        elif self.loss_id == 2:
            # Sparse Categorical Cross Entropy 
            # Non one hot encoded representation for labels
            # [2, 0]
            return tf.keras.losses.SparseCategoricalCrossentropy()
        elif self.loss_id == 3:
            # MSE
            return tf.keras.losses.MeanSquaredError()
        else:
            raise NameError('Optimizer id unknown {}'.format(self.optimizer_id))