import tensorflow as tf
import tqdm
import datetime


class Trainer(object):
    def __init__(self, config, model, optimizer, loss, train_loader, test_loader):
        self.network_id = config.network_id
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        self.use_tensorboard = getattr(config, 'use_tensorboard', False)
        if self.use_tensorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        
        self.num_epochs = config.num_epochs
        self.best_val = 0.0
    
    def load_checkpoint(self, filename):
        print('### Load_Model ###')
        print(filename)
        self.model.load_weights(filename)

    def save_checkpoint(self, filename, best_val):
        print('### Save_Model ###')
        print(filename)
        self.model.save_weights(filename)
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.criterion(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
    
    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.criterion(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def run(self):
        for epoch in range(self.num_epochs):
              # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            network_weight_path = 'models/net{net_id}_epoch{epoch_id}/'.format(net_id=str(self.network_id).zfill(4), epoch_id=str(epoch).zfill(4))
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for images, labels in tqdm.tqdm(self.train_loader):
                self.train_step(images, labels)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=(epoch + 1))
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=(epoch + 1))

            for images, labels in tqdm.tqdm(self.test_loader):
                self.test_step(images, labels)
            
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=(epoch + 1))
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=(epoch + 1))


            self.cur_val = self.test_accuracy.result()
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.test_loss.result(),
                    self.test_accuracy.result() * 100))

            self.save_checkpoint(network_weight_path, self.cur_val.numpy() > self.best_val)
            if self.cur_val > self.best_val:
                self.best_val = self.cur_val
            