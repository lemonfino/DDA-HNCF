from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.callbacks import EarlyStopping
from MDAmodel import build_MDA
import keras
import os.path as Path
import scipy.io as sio
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

def build_model(X, input_dims, arch, nf=0.5, std=1.0, mtype='mda', epochs=80, batch_size=32):
    if mtype == 'mda':
        model = build_MDA(input_dims, arch)
    else:
        print ("### Wrong model.")
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.1)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.1)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test_noisy, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)])
    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

    return mid_model, history


# ### Main code starts here
if __name__ == "__main__":
    # Training settings
    model_type = 'mda'
    select_net1=['drug_gaussian','drug_morgan','drug_network']
    select_arch1=[2]
    epochs = 80
    batch_size = 32
    nf = 0.5
    K = 5

    arch1 = {}
    arch1['mda'] = {}
    arch1['mda'] = {0: [3 * 1200, 100, 3 * 1200],
                    1: [3 * 1200, 3 * 600, 100, 3 * 600, 3 * 1200],
                    2: [3 * 2408, 3 * 1600, 3 * 1000, 100, 3 * 1000, 3 * 1600, 3 * 2408],
                    3: [3 * 1200, 3 * 800, 3 * 400, 3 * 100, 100, 3 * 100, 3 * 400, 3 * 800, 3 * 1200],
                    4: [3 * 1200, 3 * 900, 3 * 600, 3 * 300, 3 * 100, 100, 3 * 100, 3 * 300, 3 * 600, 3 * 900, 3 * 1200]}
    # load  matrices
    Nets= []
    input_dims = []
    for i in select_net1:
        Net = np.loadtxt('D:/DDA-HNCF/MDA/' + 'sim_' + str(i) + '.txt')
        Nets.append(Net)
        input_dims.append(Net.shape[1])

    model_names = []
    for a in select_arch1:
        print ("### [%s] Running for architecture: %s" % (model_type, str(arch1[model_type][a])))
        mid_model, history = build_model(Nets, input_dims, arch1[model_type][a], nf, 1.0, model_type, epochs, batch_size)

        feature1 = mid_model.predict(Nets)
        print(feature1.shape[1])
        np.savetxt('D:/DDA-HNCF/MDA/drugfeature1.txt', feature1)

if __name__ == "__main__":
    # Training settings
    model_type = 'mda'
    select_net2 = ['dis_gaussian','dis_mesh','dis_network','dis_symptom']
    select_arch2=[2]
    epochs = 80
    batch_size = 32
    nf = 0.5
    K = 5

    arch2 = {}
    arch2['mda'] = {}
    arch2['mda'] = {0: [4 * 76, 100, 4 * 76],
                    1: [4 * 76, 4 * 50, 100, 4 * 50, 4 * 76],
                    2: [4 * 2092, 4 * 1600, 4 * 1000, 100, 4 * 1000, 4 * 1600, 4 * 2092],
                    3: [4 * 76, 4 * 50, 4 * 40, 4 * 30, 100, 4 * 30, 4 * 40, 4 * 50, 4 * 76],
                    4: [4 * 76, 4 * 60, 4 * 50, 4 * 40, 4 * 30, 100, 4 * 30, 4 * 40, 4 * 50, 4 * 60, 4 * 76]}
    # load  matrices
    nets = []
    input_dim = []
    for i in select_net2:
        net= np.loadtxt('D:/DDA-HNCF/MDA/' + 'sim_' + str(i) + '.txt')
        nets.append(net)
        input_dim.append(net.shape[1])

    model_names = []
    for a in select_arch2:
        print ("### [%s] Running for architecture: %s" % (model_type, str(arch2[model_type][a])))
        mid_model, history = build_model(nets, input_dim, arch2[model_type][a], nf, 1.0, model_type, epochs, batch_size)

        feature2 = mid_model.predict(nets)
        print(feature2.shape[1])
        np.savetxt('D:/DDA-HNCF/MDA/disfeature1.txt', feature2)





