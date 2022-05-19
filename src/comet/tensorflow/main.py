"""
comet/tensorflow/main.py

Contains simple example demonstrating how to integrate comet-ml with TensorFlow
"""
from __future__ import absolute_import, division, print_function, annotations
import logging
import hydra

import comet_ml
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from omegaconf import DictConfig


log = logging.getLogger(__name__)


TF_FLOAT = tf.keras.backend.floatx()


class Net(Model):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        return self.d2(self.d1(self.flatten(self.conv1(x))))


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.net = Net()
        self.data = self.setup_data()
        self.experiment = comet_ml.Experiment(
            api_key='r7rKFO35BJuaY3KT1Tpj4adco',
            project_name='MLOps',
            auto_histogram_gradient_logging=True,
        )
        self.experiment.log_parameters(cfg)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_acc'
        )

        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_acc'
        )
        self.writer = tf.summary.create_file_writer('./outputs')
        self.net.compile(optimizer='adam',
                         loss=self.loss_obj,
                         metrics=['accuracy'])

    def setup_data(self):
        mnist = tf.keras.datasets.mnist
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        xtrain, xtest = xtrain / 255.0, xtest / 255.0
        xtrain = xtrain[..., None].astype(TF_FLOAT)
        xtest = xtest[..., None].astype(TF_FLOAT)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (xtrain, ytrain)
        ).shuffle(10000).batch(self.cfg.batch_size)

        test_ds = tf.data.Dataset.from_tensor_slices(
            (xtest, ytest)
        ).batch(self.cfg.batch_size)
        
        return {
            'train': {
                'x': xtrain,
                'y': ytrain,
                'data': train_ds,
            },
            'test': {
                'x': xtest,
                'y': ytest,
                'data': test_ds,
            },
        }

    @tf.function
    def train_step(self, images, labels) -> None:
        with tf.GradientTape() as tape:
            predictions = self.net(images, training=True)
            loss = self.loss_obj(labels, predictions)
        grads = tape.gradient(loss, self.net.trainable_variables)
        updates = zip(grads, self.net.trainable_variables)
        self.optimizer.apply_gradients(updates)

        self.train_loss(loss)
        self.train_acc(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.net(images, training=False)
        tloss = self.loss_obj(labels, predictions)
        self.test_loss(tloss)
        self.test_acc(labels, predictions)

    def train(self):
        for epoch in range(self.cfg.num_epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_acc.reset_states()
            self.test_acc.reset_states()
            for images, labels in self.data['train']['data']:
                self.train_step(images, labels)

            for timages, tlabels in self.data['test']['data']:
                self.test_step(timages, tlabels)

            if epoch % self.cfg.logfreq == 0:
                metrics = {
                    'train': {
                        'loss': self.train_loss.result(),
                        'acc': self.train_acc.result(),
                    },
                    'test': {
                        'loss': self.test_loss.result(),
                        'acc': self.test_acc.result(),
                    }
                }
                train_str = ['TRAIN:'] + [
                    f'{k}: {v.numpy():.4f}'
                    for k, v in metrics['train'].items()
                ]
                test_str = ['TEST:'] + [
                    f'{k}: {v.numpy():.4f}'
                    for k, v in metrics['train'].items()
                ]
                log.info(' '.join([
                    f'EPOCH: {epoch}',
                    *train_str,
                    *test_str,
                ]))

                with self.writer.as_default():
                    for type, data in metrics.items():
                        for key, val in data.items():
                            name = f'{type}/{key}'
                            tf.summary.scalar(
                                f'{type}/{key}',
                                val,
                                step=epoch
                            )




@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.net.fit(
    #     trainer.data['train']['x'],
    #     trainer.data['train']['y'],
    #     batch_size=cfg.batch_size,
    #     epochs=cfg.num_epochs,
    #     validation_data=(
    #         trainer.data['test']['x'],
    #         trainer.data['test']['y']
    #     ),
    # )

    # score = trainer.net.evaluate(
    #     trainer.data['test']['x'],
    #     trainer.data['test']['y']
    # )

    # log.info(f'Score: {score}')



if __name__ == '__main__':
    main()




