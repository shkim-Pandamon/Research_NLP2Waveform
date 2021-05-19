from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import json

class ResultEvaluater(object):
    def __init__(self):
        pass

    def evaluate(self, labels, preds, save_path):
        ## pd classification
        pd_idx = np.where(labels != 4)[0]
        t_labels = labels[pd_idx]
        t_preds = preds[pd_idx]
        t_idx = np.where(t_preds != 4)[0]
        pc_labels = labels[t_idx]
        pc_preds = preds[t_idx]
        cm = self.cm_maker(
            y_true = pc_labels,
            y_pred = pc_preds
        )
        self.plot_confusion_matrix(
            cm = cm,
            plot_path = save_path + '_cm_pd_classification.png',
            target_names = ["Corona", "Floating", "Particle", "Void"],
            title = "result is "
        )

    def cm_maker(self, y_true, y_pred):
        num_class = np.max(y_true)
        cm = np.zeros((num_class + 1, num_class + 1))
        for jj in range(len(y_true)):
            true = y_true[jj]
            pred = y_pred[jj]
            cm[true, pred] += 1
        cm = cm.astype("int")
        return cm
    
    def plot_confusion_matrix(self, cm, plot_path, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        accuracy = np.trace(cm_norm) / len(cm_norm)
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')
        if normalize:
            cm = cm_norm

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        if title is not None:
            plt.title(title + '\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.colorbar()

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)
        
        if labels:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_path)