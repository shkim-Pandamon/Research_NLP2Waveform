import numpy as np
import glob
import wfdb
import pickle
import ray

## Set up
seg_len_limit = 1250
target = ['ABP', 'PLETH', 'I']
# save_path = '/git_clones/Research_NLP2Waveform/basic_study_tf/BP/pickled_data/'
save_path = './basic_study_tf/BP/abp_ppg_i/'

@ray.remote
def run_patient(patient):
    print(patient)
    segment_list = glob.glob(patient + '/*.hea')
    master_list = glob.glob(patient + '/p*.hea')
    segment_list = [item for item in segment_list if item not in master_list]
    for ii, segment in enumerate(segment_list):
        try:
            tdata = {}
            signals, fields = wfdb.rdsamp(segment[:-4])
            if (fields["sig_len"] >seg_len_limit) & np.prod([(dtype in fields["sig_name"]) for dtype in target]):
                signals = np.transpose(signals)
                for dtype in target:
                    idx = np.where(np.asarray(fields["sig_name"]) == dtype)[0]
                    tdata[dtype] = signals[idx].squeeze().tolist()
                data_name = save_path + segment.split("/")[-2] + "_" + segment.split("/")[-1][:-4]
                print('data_ready')
                with open(data_name + ".pickle", "wb") as pickle_file:
                    pickle.dump(tdata, pickle_file)
        except:
            pass   

#%% Main
data_path = '/data_storage/mimic_iii_waveform/physionet.org/files/mimic3wdb-matched/1.0'
groups = glob.glob(data_path + '/p*')
for group in groups[:1]:
    print(group)
    patients = glob.glob(group + '/p*')
    ray.init()
    ray.get([run_patient.remote(patient) for patient in patients])
    ray.shutdown()
    print("group finish")