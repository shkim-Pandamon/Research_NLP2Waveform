import pickle
import numpy as np

class researcher(object):
    def __init__(self):
        pass

    def load_prps(self, data_path):
        with open(data_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        return data
    
    def create_model(self):
        from func.model_creator import ModelCreator
        model = ModelCreator().create_model()
        return model

    def build_dataset(self, data, batch_size):
        import numpy as np
        import tensorflow as tf
        from tensorflow.data import Dataset
        dataset = {}
        train_dataset = self.make_batch(data["train_x"], data["train_y"], batch_size, 100)
        validation_dataset = self.make_batch(data["validation_x"], data["validation_y"], batch_size, 5)
        dataset["train"] = Dataset.from_tensor_slices(train_dataset)
        dataset["validation"] = Dataset.from_tensor_slices(validation_dataset)        
        data["test_y"] = tf.one_hot(data["test_y"], 4)
        test = {
            "test_x": data["test_x"],
            "test_y": data["test_y"]
        }
        return dataset, test
    
    def make_batch(self, x, y, batch_size, step_size):
        labels = np.unique(y)
        step_x = np.zeros([step_size, len(labels)*batch_size, 3600, 128])
        step_y = np.zeros([step_size, len(labels)*batch_size, 4])
        batch_x = np.zeros([len(labels)*batch_size, 3600, 128])
        batch_y = np.zeros([len(labels)*batch_size, 4])
        for step in range(step_size):
            for ii in range(len(labels)):
                idx = np.where(y == ii)[0]
                np.random.shuffle(idx)
                tg_idx = idx[:batch_size]
                batch_x[ii*batch_size:(ii+1)*batch_size] = x[tg_idx]
                batch_y[ii*batch_size:(ii+1)*batch_size, ii] = 1
            step_x[step] = batch_x
            step_y[step] = batch_y
        return (step_x, step_y)

    def train_model(self, model, data, epoch):
        import tensorflow as tf
        train = data["train"]
        validation = data["validation"]
        history = model.fit(
            train,
            epochs = epoch,
            validation_data = validation
        )
        return history

    def predict_model(self, model, test, save_path):
        test_x = test["test_x"]
        test_y = np.argmax(test["test_y"], axis = 1)
        pred_y = model.predict(test["test_x"])
        pred_y = np.argmax(pred_y, axis = 1)
        from func.result_evaluator import ResultEvaluater
        ResultEvaluater().evaluate(test_y, pred_y, save_path)
        return test_y, pred_y

#%%
if __name__ == "__main__":
    sooho = researcher()
    dataset = sooho.load_prps(data_path = "/data_storage/GIS_PD/prps622pd.pickle")
    model = sooho.create_model()
    dataset, test = sooho.build_dataset(data = dataset, batch_size = 8)
    history = sooho.train_model(model = model, data = dataset, epoch = 50)
    test_y, pred_y = sooho.predict_model(model = model, test = test, save_path = "./basic_study_tf/GISPD/result")
    print("Done")
