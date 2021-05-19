import numpy as np 
import pickle
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
data = []
data_list = glob.glob('./basic_study_tf/BP/abp_ppg_i/*.pickle')[:100]
for data_path in data_list[2:]:
    tdata = pickle.load(open(data_path, "rb"))
    fig = plt.figure()
    plt.plot(tdata['ABP'][300:1300])
    plt.title('ABP')
    plt.savefig('ABP.png')

    fig = plt.figure()
    plt.plot(tdata['PLETH'][300:1300])
    plt.title('PLETH')
    plt.savefig('PLETH.png')

    fig = plt.figure()
    plt.plot(tdata['I'][300:1300])
    plt.title('I')
    plt.savefig('I.png')

    print("hi")