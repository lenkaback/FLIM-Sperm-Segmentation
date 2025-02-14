#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse
import Metrics
import shutil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
import glob
import random


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--l2", default=0.001, type=float, help="Random seed")
parser.add_argument("--dropout", default=0, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--weight_mid", default=0, type=int, help="Random seed")
# parser.add_argument("--weight_back", default=1, type=int, help="Random seed")
parser.add_argument("--weight_head", default=1, type=int, help="Random seed")
parser.add_argument("--size", default=1, type=float)
parser.add_argument("--name", default=1)
parser.add_argument("--reduce_factor", default=0.5, type=float)
parser.add_argument("--epochs", default=150, type=int)


# example run in CL with: 'python train.py --learning_rate 0.001 --l2 0.001 --weight_mid 0 --weight_head 1 --reduce_factor 0.5 --epochs 150'
def main(args):
    L2 = args.l2
    learning_rate = args.learning_rate
    weigth_head = args.weight_head
    weigth_mid = args.weight_mid
    # weight_back = args.weight_back

    TRAIN_LENGTH = 60 #info.splits['train'].num_examples # find what it is for me
    BATCH_SIZE = 32
    BUFFER_SIZE = 100
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    EPOCHS = 20

    HEIGHT, WIDTH = 256, 256
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 3

    files = [os.path.join("Trn", os.path.basename(x)) for x in glob.glob("./Dataset/Masks/Midpieces/Trn/*.tif")]

    random.shuffle(files)
    files = files[:int(len(files)*args.size)]
    test_files = [os.path.join("Val", os.path.basename(x)) for x in glob.glob("./Dataset/Masks/Midpieces/Val/*.tif")]

    train_dataset = tf.data.Dataset.from_tensor_slices(files)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_files)

    def fc(x):
        image_fn = x.numpy().decode()
        #print(image_fn)
        input_dir, input_heads, input_mids = os.path.join("./Dataset/Images", image_fn), os.path.join("./Dataset/Masks/Heads", image_fn), os.path.join("./Dataset/Masks/Midpieces", image_fn)
        # some complex pil and numpy work nothing to do with tf
        input_image = np.array(Image.open(input_dir))
        input_head = np.array(Image.open(input_heads)) / 255
        input_mid = np.array(Image.open(input_mids)) / 255

        input_mid[input_head==1] = 0
        input_mask = 2*input_head + input_mid
        input_mask = input_mask[:,:,tf.newaxis]

        return input_image, input_mask
        #return np.zeros([30,60,3], np.uint8), np.ones([30,60,3], np.uint8)

    def add_sample_weight(image, label):
        sample_weighted_label = weigth_mid*tf.cast(label==1, tf.float32) + 1*tf.cast(label!=2, tf.float32) + weigth_head*tf.cast(label==2, tf.float32)
    #     sample_weighted_label = weigth_mid*tf.cast(label!=0, tf.float32) + 2 + weigth_head*tf.cast(label==2, tf.float32)
        return image, label, sample_weighted_label

    def shear(image, label):
        con = np.concatenate((image.numpy(), label.numpy()), axis=-1)
        sheared = tf.keras.preprocessing.image.random_shear(
                    con, intensity=20, row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect')
        #return tf.convert_to_tensor(rot[:,:,:-1], dtype=tf.float32), tf.convert_to_tensor(rot[:,:,-1], dtype=tf.float32)
        sh_img, sh_mask = sheared[:, :, :-1], sheared[:,:,-1]
        return sh_img, sh_mask[:, :, tf.newaxis]

    def wrapper_shear(img, label):
        x, y = tf.py_function(shear, inp=(img, label), Tout=(tf.uint8, tf.uint8))
        x.set_shape([512,512,INPUT_CHANNELS])
        y.set_shape([512,512, 1])
        return x, y

    def rotate(image, label):
        con = np.concatenate((image.numpy(), label.numpy()), axis=-1)
        rot = tf.keras.preprocessing.image.random_rotation(
                    con, 90, row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect', interpolation_order=1
                    )
        #return tf.convert_to_tensor(rot[:,:,:-1], dtype=tf.float32), tf.convert_to_tensor(rot[:,:,-1], dtype=tf.float32)
        rot_img, rot_mask = rot[:,:,:-1], rot[:,:,-1]
        return rot_img, rot_mask[:,:,tf.newaxis]

    def wrapper_rot(img, label):
        x, y = tf.py_function(rotate, inp=(img, label), Tout=(tf.uint8, tf.uint8))
        x.set_shape([512,512,INPUT_CHANNELS])
        y.set_shape([512,512, 1])
        return x, y

    def add_noise(img, label):
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32)
        noise_img = tf.add(img, noise)
        return noise_img, label

    def wrapper(fn):
        x,y= tf.py_function(fc, inp=(fn,), Tout=(tf.uint8, tf.uint8))
        x.set_shape([512,512,INPUT_CHANNELS])
        y.set_shape([512,512, 1])
        return x, y

    def flip(img, label):
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            label = tf.image.flip_left_right(label)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            label = tf.image.flip_up_down(label)
        if tf.random.uniform(()) > 0.5:
            img = tf.transpose(img, perm=[1, 0, 2])
            label = tf.transpose(label, perm=[1, 0, 2])
        #label = label[:,:, 0]
        #label.set_shape([HEIGHT,WIDTH])
        return img, label

    def normalize(img, label):
        img = tf.cast(img, tf.float32) / 255. #tf.cast(tf.math.reduce_max(input_image), tf.float32)
        #input_mask -= 1, make sure the mask is from {0, 1, 2}
    #     img.set_shape([HEIGHT,WIDTH,INPUT_CHANNELS])
    #     label.set_shape([HEIGHT,WIDTH, 1])
        return img, tf.cast(label, tf.float32)

    def random_crop(img, label):
        height = 512
        width = 512
        input_height = HEIGHT
        input_width = WIDTH
        start_h = tf.random.uniform([], 0, height-input_height, dtype=tf.int32)
        start_w = tf.random.uniform([], 0, width-input_width, dtype=tf.int32)
        img = img[start_h:start_h+input_height, start_w:start_w+input_width]
        label = label[start_h:start_h+input_height, start_w:start_w+input_width]
        return img, label

    train_dataset = train_dataset.shuffle(len(files)).map(wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(wrapper_rot, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(random_crop).map(flip).map(normalize).map(add_sample_weight)
    train_dataset = train_dataset.repeat(10).batch(32,drop_remainder=True).prefetch(3)
    test_dataset = test_dataset.map(wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(normalize)
    test_dataset = test_dataset.batch(16)#.prefetch(1)


    inputs = Input((None, None, INPUT_CHANNELS))
    l2 = tf.keras.regularizers.L2(L2)
    #s = Lambda(lambda x: x / 255) (inputs)
    dropout = 0

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    #c1 = Dropout(dropout) (c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c1)
    c1 = BatchNormalization()(c1)
    #p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Conv2D(32, (3, 3), strides=(2, 2), activation=None, kernel_initializer='he_normal', kernel_regularizer=l2, padding='same')(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (p1)
    c2 = BatchNormalization()(c2)
    #c2 = Dropout(dropout) (c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = Conv2D(64, (3, 3), strides=(2, 2), activation=None, kernel_initializer='he_normal', kernel_regularizer=l2, padding='same')(c2)
    #p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (p2)
    c3 = BatchNormalization()(c3)
    #c3 = Dropout(dropout) (c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = Conv2D(128, (3, 3), strides=(2, 2), activation=None, kernel_initializer='he_normal', kernel_regularizer=l2, padding='same')(c3)
    #p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (p3)
    c4 = BatchNormalization()(c4)
    #c4 = Dropout(dropout) (c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = Conv2D(256, (3, 3), strides=(2, 2), activation=None, kernel_initializer='he_normal', kernel_regularizer=l2, padding='same')(c4)
    #p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (p4)
    c5 = BatchNormalization()(c5)
    #c5 = Dropout(dropout) (c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c5)
    c5 = BatchNormalization()(c5)
    p5 = Conv2D(512, (3, 3), strides=(2, 2), activation=None, kernel_initializer='he_normal', kernel_regularizer=l2, padding='same')(c5)

    cx = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (p5)
    cx = BatchNormalization()(cx)
    #c5 = Dropout(dropout) (c5)
    cx = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (cx)
    cx = BatchNormalization()(cx)
    cx = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (cx)
    cx = BatchNormalization()(cx)

    uy = Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_regularizer=l2, padding='same') (cx)
    uy = concatenate([uy, c5])
    cy = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (uy)
    cy = BatchNormalization()(cy)
    cy = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (uy)
    cy = BatchNormalization()(cy)
    #cy = Dropout(dropout) (cy)
    cy = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (cy)
    cy = BatchNormalization()(cy)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_regularizer=l2, padding='same') (cy)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u6)
    c6 = BatchNormalization()(c6)
    #c6 = Dropout(dropout) (c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2,) (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u7)
    c7 = BatchNormalization()(c7)
    #c7 = Dropout(args.dropout) (c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2,) (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u8)
    c8 = BatchNormalization()(c8)
    #c8 = Dropout(dropout) (c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2,) (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (u9)
    c9 = BatchNormalization()(c9)
    #c9 = Dropout(dropout) (c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2, padding='same') (c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask

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

    def dice_coef_3cat(y_true, y_pred, smooth=1e-7):
        '''
        Dice coefficient for 10 categories. Ignores background pixel label 0
        Pass to model as metric during compile statement
        '''
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))

    def dice_coef_3cat_loss(y_true, y_pred):
        '''
        Dice loss to minimize. Pass to model as loss during compile statement
        '''
        return 1 - dice_coef_3cat(y_true, y_pred)

    def loss(y_true, y_pred):
        sample_weight = weigth_mid * tf.cast(y_true == 1, tf.float32) + 1 * tf.cast(y_true != 2, tf.float32) + weigth_head * tf.cast(y_true == 2, tf.float32)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return scce(y_true, y_pred, sample_weight)# + dice_coef_3cat_loss(y_true, y_pred)

    model = Model(inputs=[inputs], outputs=[outputs])

    train = []
    test = []
    lr = []
    e = []

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', UpdatedMeanIoU()]
    )

    model_name = None
    best = 0
    best_epoch = -1

    epochs = args.epochs + 1
    for epoch in range(epochs):
        print(f"Epoch {epoch}:\n")
        for batch_id, (x, y, s) in enumerate(train_dataset):
            loss_metrics = model.train_on_batch(x, y, s, reset_metrics=True, return_dict=False)
        print(f"Train loss after {batch_id}/{len(train_dataset)}: {loss_metrics[0]} - Train Acc {loss_metrics[1]} - Train IoU {loss_metrics[-1]}")
        for x, y in test_dataset:
            loss_metrics = model.test_on_batch(x, y)
        print(f"Val loss {loss_metrics[0]} - Val Acc {loss_metrics[1]} - Val IoU {loss_metrics[-1]}\n")
        if epoch != 0 and epoch % 25 == 0:
            rf = float(args.reduce_factor)
            model.optimizer.learning_rate = model.optimizer.learning_rate * rf
            print(f"Reducing LR to {model.optimizer.learning_rate}")
        if epoch % 10 == 0:
            pred_masks = []
            gt_masks = []
            for i, t, _ in train_dataset:
                pred_masks.append(np.array(create_mask(model.predict(i))))
                gt_masks.append(np.array(t))

            pred_masks = np.array(pred_masks)[0]
            gt_masks = np.array(gt_masks)[0]

            m = Metrics.Metrics(gt_masks, pred_masks, iou_thresh=0.4, verbal=False)
            print("\n Metrics:\n")
            print(m.df_metrics)
            train.append(m.df_metrics)
            print("\n")

            lr.append(model.optimizer.learning_rate)
            e.append(epoch)

            pred_masks = []
            gt_masks = []
            for i, t in test_dataset:
                pred_masks.append(np.array(create_mask(model.predict(i))))
                gt_masks.append(np.array(t))

            pred_masks = np.array(pred_masks)[0]
            gt_masks = np.array(gt_masks)[0]

            m = Metrics.Metrics(gt_masks, pred_masks, iou_thresh=0.4, verbal=False)
            print("\n Metrics:\n")
            print(m.df_metrics)
            test.append(m.df_metrics)
            print("\n")

            if best < test[-1].at["All", "F1-score"]:
                best = test[-1].at["All", "F1-score"]
                best_epoch = epoch
                if model_name != None:
                    shutil.rmtree(model_name)
                model_name = f"./Models/Model_{best:.2f}_h{args.weight_head}_m{args.weight_mid}_lr{args.learning_rate}_l2{args.l2}_red{args.reduce_factor}_eps{best_epoch}"
                model.save(model_name)
    x = e

    names = ["Mito", "Head", "All"]
    t_names = ["Train", "Test"]
    train_test = [train, test]

    import matplotlib.pyplot as plt
    from matplotlib import pyplot as plt

    for j in range(2):
        metric = train_test[j]
        fig, axs = plt.subplots(1, 3, figsize=(40, 10))
        fig.suptitle(f'{t_names[j]}')

        for i in range(3):
            acc, rec, prec, f1 = [], [], [], []
            for k in range(len(metric)):
                acc.append(metric[k].at[names[i], "Accuracy"])
                rec.append(metric[k].at[names[i], "Recall"])
                prec.append(metric[k].at[names[i], "Precision"])
                f1.append(metric[k].at[names[i], "F1-score"])

            axs[i].plot(x, acc, c='k', marker='.', label='Accuracy')
            axs[i].plot(x, rec, c='b', marker='x', label='Recall')
            axs[i].plot(x, prec, c='g', marker='s', label='Precision')
            axs[i].plot(x, f1, c='r', marker='o', label='F1-Score')
            axs[i].set_title(f"{names[i]}")
            axs[i].axis([0, epochs, 0, 100])

        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.savefig(f"{model_name}/{t_names[j]}.png")

    f = open(f"{model_name}/metrics.txt", "w")
    for i in range(len(train)):
        if best_epoch == i:
            f.write(f"Best Epoch {e[i]}\n TRAIN: \n")
        else:
            f.write(f"Epoch {e[i]}\n TRAIN: \n")
        f.write(train[i].to_string())
        f.write("\n\n")
        f.write(f"TEST: \n")
        f.write(test[i].to_string())
        f.write("\n\n")
    f.close()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
