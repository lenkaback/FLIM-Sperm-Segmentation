import numpy as np
import skimage.measure
import skimage.morphology
import glob
import tifffile
import tensorflow as tf

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

folder = "../testimg"
tiff_files = glob.glob(f"{folder}/Tiffs/*.tif")
print(tiff_files)
tiffs = []

mod = "../model"

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

    file = t.split("/")[-1]
    file = file.split(".")[0]
    path = f"{folder}/Export"

    for i in [1, 2]:
        mask = pred_mask == i
        type = "Head" if i == 2 else "Mid"
        n = f"{path}/Not_sep/{type}/{file}.tif"
        print(n)

        try:
            os.makedirs(f"{path}/Not_sep/{type}/", exist_ok=True)
        except OSError as error:
            print(error)

        ms = 50 if i == 2 else 100
        mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=ms).astype(np.uint8)

        tifffile.imwrite(n, mask.astype(np.uint8)*255)

        mask, n_detected = skimage.measure.label(mask, background=0, return_num=True, connectivity=1)

        for j in np.unique(mask):
            if j == 0:
                continue
            name = f"{path}/{type}/{file}/{file}_{j}.tif"

            try:
                os.makedirs(f"{path}/{type}/{file}", exist_ok=True)
            except OSError as error:
                print(error)

            tifffile.imwrite(name, (mask == j).astype(np.uint8) * 255)