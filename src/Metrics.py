import numpy as np
import skimage.measure
import skimage.morphology
import scipy.optimize
from PIL import Image
import matplotlib.pyplot as plt
import glob
import pandas as pd

class Metrics:
    def __init__(self, gt, pred, iou_thresh=0.5, file_names=None, verbal=False):
        self.pred = pred
        self.gt = gt
        self.IoU_thresh = iou_thresh
        self.file_names = file_names
        self.verbal = verbal
        self.stats = self.CalculateStats()
        self.metrics = self.CalculateMetrics()
        self.df_metrics = pd.DataFrame(data=self.metrics, index=["Mito", "Head", "All"],
                                  columns=["IoU", "Accuracy", "Recall", "Precision", "F1-score"])
        self.df_stats = pd.DataFrame(data=self.stats, index=["Mito", "Head", "All"],
                                columns=["TP", "Detected", "GT"], dtype=np.uint32)
        if self.verbal:
            self.print()

    def CalculateStats(self):
        stats = np.empty((3, 3))
        for i in (1, 2):
            stats[i-1] = self.calc_stats_obj(self.pred == i, self.gt == i, i)
        stats[-1] = np.sum(stats[0:2], axis=0)
        return stats

    def calc_stats_obj(self, pred_masks, gt_masks, m_h):
        true_false = []
        qp = 0
        size = 50 if m_h == 2 else 100

        for p, g in zip(pred_masks, gt_masks):
            pred = skimage.morphology.remove_small_objects(p.astype(bool), min_size=size).astype(np.uint8)
            g = skimage.morphology.remove_small_objects(g.astype(bool), min_size=size).astype(np.uint8)
            pred, n_detected = skimage.measure.label(pred, background=0, return_num=True, connectivity=1)
            gt, n_gt = skimage.measure.label(g, background=0, return_num=True, connectivity=1)

            distances = np.full((n_detected + 1, n_gt + 1), np.finfo(np.float64).max, np.float)

            for i in np.unique(pred):
                pred_object = pred == i
                overlayed_gt_objects = np.unique(gt[pred_object])
                #             print("pred obj", i)
                for j in overlayed_gt_objects:
                    gt_object = gt == j
                    intersection = np.logical_and(gt_object, pred_object).sum()
                    union = np.logical_or(gt_object, pred_object).sum()
                    IoU = intersection / union
                    #                 print("\tgt obj", j, "iou", f"{IoU*100:.1f}%")
                    distances[i, j] = 1 - IoU if IoU > self.IoU_thresh else np.finfo(np.float64).max

                    d = np.logical_xor(gt_object, pred_object)
                    if pred_object[d > 0].sum() == 0 or gt_object[d > 0].sum() == 0:
                        distances[i, j] = 0

            r, c = scipy.optimize.linear_sum_assignment(distances)
            couples = []
            for row, column in zip(r, c):
                value = distances[row, column]

                if value < np.finfo(np.float64).max:
                    if row * column == 0:
                        pass
                    else:
                        #print('(%d, %d) -> %d' % (row, column, value))
                        couples.append((row, column))

            #print(couples)
            if self.file_names != None:
                self.export(couples, pred, gt, qp, m_h)

            n_tp = len(couples)
            true_false.append((n_tp, n_detected, n_gt))
            qp += 1
        return np.sum(np.array(true_false), axis=0)

    def CalculateMetrics(self):
        metrics = np.empty((3, 5))
        for i in range(2):
            metrics[i] = np.concatenate((Metrics.calc_img(self.gt == i+1, self.pred == i+1),
                                        Metrics.calc_stats(*self.stats[i])))
        metrics[-1] = np.concatenate((Metrics.calc_img(self.gt, self.pred),
                                    Metrics.calc_stats(*self.stats[-1])))
        return metrics*100

    def print(self):
        print(self.df_metrics)
        print(self.df_stats)

    @staticmethod
    def calc_img(gt, pred):
        return [Metrics.IoU(gt, pred), Metrics.accuracy(gt, pred)]

    @staticmethod
    def calc_stats(n_tp, n_detected, n_gt):
        recall, precision = Metrics.recall(n_tp, n_gt), Metrics.precision(n_tp, n_detected)
        return [recall, precision, Metrics.f1(recall, precision)]

    @staticmethod
    def recall(n_tp, n_gt):
        return n_tp / n_gt

    @staticmethod
    def precision(n_tp, n_detected):
        return 0. if n_detected == 0 else n_tp / n_detected

    @staticmethod
    def f1(recall, precision):
        if precision == 0 or recall == 0:
            return 0
        return 2 * ((recall*precision)/ (recall+precision))

    @staticmethod
    def IoU(gt, pred):
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        return intersection / union

    @staticmethod
    def accuracy(gt, pred):
        return np.mean(gt == pred)

    def get_stats(self):
        return self.stats

    def get_metrics(self):
        return self.metrics