import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

import numpy as np
import h5py as h5
import os

from models import SFANet
from utils.preprocess import *

class DataGenerator(Sequence):
    def __init__(self, f, batch_size):
        self.train_images = f['train/images']
        self.train_confidence = f['train/confidence']
        self.train_attention = f['train/attention']
        self.batch_size = batch_size
        self.indices = np.arange(len(self.train_images))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.train_images) // self.batch_size

    def __getitem__(self, index):
        batch_inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = np.stack([self.train_images[i] for i in batch_inds])
        batch_confidence = np.stack([self.train_confidence[i] for i in batch_inds])
        batch_attention = np.stack([self.train_attention[i] for i in batch_inds])
        return batch_images, (batch_confidence, batch_attention)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def main():
    # 数据集路径和日志目录
    data_path = r"F:\OneDrive - PennO365\Project\urban-tree-detection\h5\output.h5"
    log_dir = r"F:\OneDrive - PennO365\Project\urban-tree-detection\logs"

    lr = 1e-4
    epochs = 500
    batch_size = 8

    # GPU configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

    # Load data
    f = h5.File(data_path, 'r')
    bands = f.attrs['bands']
    val_images = f['val/images'][:]
    val_confidence = f['val/confidence'][:]
    val_attention = f['val/attention'][:]

    preprocess_fn = eval(f'preprocess_{bands}')

    # Build model
    model, testing_model = SFANet.build_model(
        val_images.shape[1:],
        preprocess_fn=preprocess_fn
    )
    opt = Adam(lr)
    model.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], loss_weights=[1, 0.1])

    print(model.summary())

    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    callbacks = []
    weights_best_path = os.path.join(log_dir, 'weights.best.h5')
    callbacks.append(ModelCheckpoint(
        filepath=weights_best_path,
        monitor='val_loss',
        verbose=True,
        save_best_only=True,
        save_weights_only=True,
    ))
    weights_latest_path = os.path.join(log_dir, 'weights.latest.h5')
    callbacks.append(ModelCheckpoint(
        filepath=weights_latest_path,
        monitor='val_loss',
        verbose=True,
        save_best_only=False,
        save_weights_only=True,
    ))
    tensorboard_path = os.path.join(log_dir, 'tensorboard')
    os.system(f"rm -rf {tensorboard_path}")
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))

    # Use DataGenerator
    train_gen = DataGenerator(f, batch_size)
    y_val = (val_confidence, val_attention)

    # Train model
    model.fit(
        train_gen,
        validation_data=(val_images, y_val),
        epochs=epochs,
        verbose=True,
        callbacks=callbacks
    )

if __name__ == '__main__':
    main()
