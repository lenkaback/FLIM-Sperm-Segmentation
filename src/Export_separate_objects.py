import numpy as np
import skimage.measure
import skimage.morphology
import scipy.optimize
from PIL import Image
import matplotlib.pyplot as plt
import glob
import tifffile
import tensorflow as tf
from skimage import io
import sys

from tensorflow.keras.models import Model, load_model
import os

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=3,
               name=None,
               dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

folder = ".\\Unanotated\\2022_NADH_2-DOGtreat\\20220212_NADH_2-DOG\\20220214_NADH_2-DOG\\61389\\Ctrl_HBSS\\"
tiff_files = glob.glob(f"{folder}*.tif")
print(tiff_files)
tiffs = []

mod = "Model_Final\\Model_80.32_h3_m0_lr0.0005_l20.001_red0.5_name8_eps80\\"

model = load_model(mod, compile=False, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU()})

for i in range(len(tiff_files)):
    t = tiff_files[i]
    print(t)
    im = tifffile.imread(t)
    im = im[np.newaxis, ...]/255.
    print(im.shape)

    pred_mask = np.array(create_mask(model.predict(im)))
    print(pred_mask.shape)
    pred_mask = np.array(pred_mask)[0, :, :, 0]

    file = t.split("\\")[-1]
    file = file.split(".")[0]
    path = f"{folder}Export"

    for i in [1, 2]:
        mask = pred_mask == i
        type = "Head" if i == 2 else "Mid"
        n = f"{path}\\Not_sep\\{type}\\{file}.tif"

        try:
            os.makedirs(f"{path}\\Not_sep\\{type}\\", exist_ok=True)
        except OSError as error:
            print(error)

        ms = 50 if i == 2 else 100
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=ms).astype(np.uint8)

        Image.fromarray((mask).astype(np.uint8) * 255).save(n)

        mask, n_detected = skimage.measure.label(mask, background=0, return_num=True, connectivity=1)

        for j in np.unique(mask):
            if j == 0:
                continue
            name = f"{path}\\{type}\\{file}\\{file}_{j}.tif"

            try:
                os.makedirs(f"{path}\\{type}\\{file}", exist_ok=True)
            except OSError as error:
                print(error)

            obj = Image.fromarray((mask == j).astype(np.uint8) * 255)
            obj.save(name)