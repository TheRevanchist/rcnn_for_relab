import numpy as np

class AccuracyChecker:

    def __init__(self, n_labels):
        self.confusion = np.zeros((n_labels, n_labels))
        # batch_accuracies = np.zeros((n_batches, max_epoch))

    def update(self, real, predicted):
        if predicted.ndim == 2 and real.ndim == 2:
            predicted = np.argmax(predicted, axis=1)
            real = np.argmax(real, axis=1)

        # create confusion matrix
        conf = np.zeros(self.confusion.shape)
        for r, p in zip(real, predicted):
            conf[r, p] += 1.0

        # update the total confusion matrix
        self.confusion += conf

    def accuracy(self):
        return self.confusion.trace() / self.confusion.sum()

    def precision(self):
        return self.confusion.diagonal() / self.confusion.sum(axis=1)

    def recall(self):
        return self.confusion.diagonal() / self.confusion.sum(axis=0)

    def reset(self):
        self.confusion = np.zeros(self.confusion.shape)
