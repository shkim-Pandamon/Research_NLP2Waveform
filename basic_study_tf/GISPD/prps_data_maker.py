import tensorflow as tf
import numpy as np
import glob
import pickle

#%% definition
class researcher(object):
    def __init__(self):
        pass

    def set_up_data(self, data_path):
        data = {
            "train_x": np.zeros([0, 3600, 128]),
            "train_y": np.zeros([0], dtype = int),
            "validation_x": np.zeros([0, 3600, 128]),
            "validation_y": np.zeros([0], dtype = int),
            "test_x": np.zeros([0, 3600, 128]),
            "test_y": np.zeros([0], dtype = int)
        }
        ## pd_data
        pd_mode = ["Corona", "Floating", "Particle", "Void"]
        for ii, pmode in enumerate(pd_mode):
            prps_data = np.load(data_path + "prps_USED_%s.npy"%(pmode))
            np.random.shuffle(prps_data)
            tr_idx = round(len(prps_data) * 0.6)
            va_idx = round(len(prps_data) * 0.2) + tr_idx
            train_x = prps_data[:tr_idx]
            validation_x = prps_data[tr_idx:va_idx]
            test_x = prps_data[va_idx:]
            train_y = (np.ones(len(train_x)) * ii).astype("int")
            validation_y = (np.ones(len(validation_x)) * ii).astype("int")
            test_y = (np.ones(len(test_x)) * ii).astype("int")
            data["train_x"] = np.append(data["train_x"], train_x, axis = 0)
            data["validation_x"] = np.append(data["validation_x"], validation_x, axis = 0)
            data["test_x"] = np.append(data["test_x"], test_x, axis = 0)
            data["train_y"] = np.append(data["train_y"], train_y, axis = 0)
            data["validation_y"] = np.append(data["validation_y"], validation_y, axis = 0)
            data["test_y"] = np.append(data["test_y"], test_y, axis = 0)
        # print("listify")
        # for key in data.keys():
        #     data[key] = data[key].tolist()
        print("saving")
        with open(data_path + "prps622pd.pickle", "wb") as pickle_file:
            pickle.dump(data, pickle_file)
        print("done")

        # ## noise_data
        # #used
        # noise_data = np.zeros([0, 3600, 128])
        # prps_data = np.load(data_path + "prps_USED_Noise.npy", allow_pickle=True)
        # if pmode == "Noise":
        #     for jj, prps in enumerate(prps_data):
        #         try:
        #             noise_data = np.append(noise_data, np.expand_dims(prps, axis = 0), axis = 0)
        #         except:
        #             pass
        # #Unused
        # unused_list = glob.glob(data_path + 'prps_UNUSED_*.npy')
        # for dpath in unused_list:
        #     prps_data = np.load(dpath)
        #     noise_data = np.append(noise_data, prps_data, axis = 0)
        # idx = np.arange(len(noise_data))
        # np.random.shuffle(idx)
        # tr_len = round(len(noise_data) * 0.1)
        # va_len = round(len(noise_data) * 0.1) + tr_len
        # tr_idx = idx[:tr_len]
        # va_idx = idx[tr_len:va_len]
        # ts_idx = idx[va_len:]
        # train_x = noise_data[tr_idx]
        # validation_x = noise_data[va_idx]
        # test_x = noise_data[ts_idx]
        # train_y = (np.ones(len(tr_idx)) * 4).astype("int")
        # validation_y = (np.ones(len(va_idx)) * 4).astype("int")
        # test_y = (np.ones(len(ts_idx)) * 4).astype("int")
        # data["train_x"] = np.append(data["train_x"], train_x, axis = 0)
        # data["validation_x"] = np.append(data["validation_x"], validation_x, axis = 0)
        # data["test_x"] = np.append(data["test_x"], test_x, axis = 0)
        # data["train_y"] = np.append(data["train_y"], train_y, axis = 0)
        # data["validation_y"] = np.append(data["validation_y"], validation_y, axis = 0)
        # data["test_y"] = np.append(data["test_y"], test_y, axis = 0)
        # debug = True
        
#%% main
if __name__ == "__main__":
    ## Data load
    data_path = '/data_storage/GIS_PD/'
    sooho = researcher()
    sooho.set_up_data(data_path = data_path)
