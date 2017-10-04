import monitor
import pickle

if __name__ == '__main__':
    chk = monitor.AccuracyChecker(21)
    for fname in ("info_all_images_trainval.pickle", "info_all_images_test.pickle"):
        with open(fname, 'rb') as f:
            dataset = pickle.load(f)
        for img_data in dataset:
            predicted = img_data[0]
            real = img_data[1]
            chk.update(real, predicted)

        print(chk.confusion)
        chk.reset()
