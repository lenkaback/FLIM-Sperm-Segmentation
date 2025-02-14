import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from skimage.morphology import remove_small_objects
import tensorflow as tf
import glob
import Metrics
from tensorflow.keras.models import load_model

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
    pred_mask = np.array(pred_mask[..., tf.newaxis])

    filtered_mid = remove_small_objects((pred_mask==1).astype(bool), min_size=100).astype(np.uint8)
    filtered_heads = remove_small_objects((pred_mask==2).astype(bool), min_size=50).astype(np.uint8)
    pred_mask = filtered_mid + filtered_heads * 2
    return pred_mask

def display(tiff, gt, pred):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    display_list = [tiff, gt, pred]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(np.array(display_list[i]), vmin=0, vmax=2)
        plt.axis('off')
    plt.show()

gt_head_files = glob.glob("../testimg/Heads/*.tif")
gt_mid_files = glob.glob("../testimg/Mid/*.tif")
tiff_files = glob.glob("../testimg/Tiffs/*.tif")

print(gt_head_files)
gt_heads, gt_mid, tiffs = [], [], []

for i in range(len(gt_head_files)):
    gt_heads.append(np.array(Image.open(gt_head_files[i])))
    gt_mid.append(np.array(Image.open(gt_mid_files[i])))
    tiffs.append(np.array(Image.open(tiff_files[i]))[:,:])

gt_heads, gt_mid, tiffs = np.array(gt_heads), np.array(gt_mid), np.array(tiffs)
gt = np.zeros(gt_heads.shape)
gt[gt_heads > 0] = 2
gt[gt_mid > 0] = 1
tiffs = tiffs.astype(np.float32) / 255.
print(f"Shape of images to predict: {tiffs.shape}")

model_name = "../model"

model = load_model(model_name, compile=False, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU()})
pred_masks = np.array(create_mask(model.predict(tiffs)))
pred_masks = np.array(pred_masks)[:, :, :, 0]

for p, g, t, filename in zip(pred_masks, gt, tiffs, tiff_files):
    display(t, g, p)
    name = tf.replace("Tiffs", "Predictions")
    try:
        os.makedirs(os.path.dirname(name), exist_ok=True)
    except OSError as error:
        print(error)
    tifffile.imwrite(name, (mask == j).astype(np.uint8))

m = Metrics.Metrics(gt, pred_masks, iou_thresh=0.4, verbal=True)
print(m)

