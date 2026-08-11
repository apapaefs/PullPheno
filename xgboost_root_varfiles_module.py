import numpy as np
import random
import math
import matplotlib.pyplot as plt 
# import xgboost and sklearn stuff:
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
import pickle
from tqdm.auto import tqdm

# functions to load root varfiles from HwSim
from read_root_varfiles import *
    
###############
# FUNCTIONS   #
###############

# main training function (June 12, 2025: in progress!)
def train_xgboost(signal_SM_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, seed): 

    # load signal and backgrounds
    # NOTE THAT: the weights will also be multiplied by the total cross section for the process! 

    # load signal:
    idS=0 # id number for signal
    S, LS, wS = read_ROOT_varfile(signal_SM_file, idS, xsS)
    Sweight = Luminosity * np.sum(wS)/initial_S * sig_factors # calculate total expected number of events
    print('Signal pre-efficiency=', np.sum(wS)/initial_S/xsS)
    
    # initial values for arrays used in training: 
    X = S
    L = LS
    W = wS

    Bweight = 0
    initial_NB = {}
    for bkg in Backgrounds:
        xsB=Backgrounds_xsec[(Energy, bkg)] # background cross sections (fb)
        B, LB, wB =  read_ROOT_varfile(Background_files[(Energy, bkg)], idB[bkg], Backgrounds_xsec[(Energy, bkg)])
        initial_NB[bkg] =  Luminosity * np.sum(wB)/initial_B[bkg] * bkg_factors # calculate total expected number of events in each background
        Bweight += initial_NB[bkg] # incremenet to total expected number of events
        print('Background pre-efficiency', bkg, np.sum(wB)/initial_B[bkg]/Backgrounds_xsec[(Energy, bkg)])
        # concatenate lists:
        X = X + B
        L = L + LB
        W = W + wB

    # convert to numpy arrays: 
    X = np.array(X)
    L = np.array(L)
    W = np.array(W)

    # create testing and training samples:
    print("Splitting samples into testing and training")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, L, W, test_size=0.5,random_state=seed)

    # train XGBoost model:
    print("Training the model")
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train,sample_weight=w_train, verbose=3)
    print("Done training the model")
    return model

# apply the given model
def apply_xgboost(model, signal_SM_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, seed):

    # load signal:
    idS=0 # id number for signal
    S, LS, wS = read_ROOT_varfile(signal_SM_file, idS, xsS)
    Sweight = Luminosity * np.sum(wS)/initial_S * sig_factors # calculate total expected number of events
    print('Signal pre-efficiency=', np.sum(wS)/initial_S/xsS)
    
    # initial values for arrays used in training: 
    X = S
    L = LS
    W = wS
    
    #print(model)
    Bweight = 0
    initial_NB = {}
    for bkg in Backgrounds:
        xsB=Backgrounds_xsec[(Energy, bkg)] # background cross sections (fb)
        print(Background_files[(Energy, bkg)])
        B, LB, wB =  read_ROOT_varfile(Background_files[(Energy, bkg)], idB[bkg], Backgrounds_xsec[(Energy, bkg)])
        initial_NB[bkg] =  Luminosity * np.sum(wB)/initial_B[bkg] * bkg_factors # calculate total expected number of events in each background
        Bweight += initial_NB[bkg] # incremenet to total expected number of events
        print('Background pre-efficiency', bkg, np.sum(wB)/initial_B[bkg]/Backgrounds_xsec[(Energy, bkg)])
        # concatenate lists:
        X = X + B
        L = L + LB
        W = W + wB


    # create testing and training samples:
    print("Splitting samples into testing and training")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, L, W, test_size=0.5,random_state=seed)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    # in this case signal = 0, backgrounds = i = 1, 2,...
    # (0,0): signal-as-signal -> True positive
    # (i,0): background-as-signal (mis-id) -> False positive
    confmatrix = confusion_matrix(y_test, predictions)
    print('confusion matrix:')
    print(confmatrix)
    # signal efficiency:
    total_S = 0
    for j in range(len(Backgrounds)+1):
        total_S += confmatrix[0][j]
    eff_S = confmatrix[0][0]/total_S # signal identified as signal divided by total number of signal events
    # background effiencies:
    eff_B = {}
    for bkg in Backgrounds:
        total_B = 0
        for j in range(len(Backgrounds)+1):
            total_B += confmatrix[idB[bkg]][j]
        eff_B[bkg] = confmatrix[idB[bkg]][0]/total_B
        print(bkg, confmatrix[idB[bkg]][0], total_B)

    print('Luminosity=', Luminosity)
        
    # initial cross sections into final state:
    print('Initial signal cross section=', Sweight/Luminosity)
    print('Initial background cross section=', Bweight/Luminosity)
    print('-')
    # calculate "significance"
    print('Initial significance=', Sweight/np.sqrt(Bweight))
    print('-')
    # print analysis efficiencies
    print('Signal efficiency (xgboost only)=', eff_S)
    print('Signal efficiency full=', eff_S * np.sum(wS)/initial_S/xsS)
    print('Background Efficiencies (xgboost only)=', eff_B)
    print('-')
    print('Final signal cross section=', xsS *eff_S * np.sum(wS)/initial_S )
    # calculate the number of events for the background after the analysis:
    final_NB = {}
    final_NB_total = 0
    for bkg in Backgrounds:
        final_NB[bkg] = initial_NB[bkg] * eff_B[bkg]
        #print('\tNumber of events in', bkg,final_NB[bkg], 'after analysis')
        final_NB_total += final_NB[bkg]
    print('Final background cross section=', final_NB_total/Luminosity)
    print('Final significance=', Sweight*eff_S/np.sqrt(final_NB_total))
    print('-')
    # calculate 95% C.L. limit on expected number of events: 
    S2sigma = np.sqrt(final_NB_total) * 2
    print('95% C.L. limit on number of signal events=', S2sigma)
    print('95% C.L. limit on signal cross section in given final state=', S2sigma/Luminosity, 'fb')


# save the model:
def save_model(model, filename):
    model.save_model(str(filename))
    #with open(filename,'wb') as f:
    #    pickle.dump(model,f)

# load the model:
def load_model(filename):
    model = xgb.XGBClassifier()
    model.load_model(filename)
    #with open(filename, 'rb') as f:
    #    model = pickle.load(f)
    return model

# apply the given model
def apply_xgboost_write(modelfile, signal_file, Backgrounds, Background_files, Backgrounds_xsec, xsS, initial_S, sig_factors, initial_B, idB, bkg_factors, Luminosity, Energy, seed, smeartag):
    print('loading', modelfile)
    model = xgb.XGBClassifier()
    model.load_model(modelfile)
    print('model loaded')
    
    # load signal:
    idS=0 # id number for signal
    S, LS, wS = read_ROOT_varfile(signal_file, idS, xsS)
    Sweight = Luminosity * np.sum(wS)/initial_S * sig_factors # calculate total expected number of events
    #print('Signal pre-efficiency=', np.sum(wS)/initial_S/xsS)
    
    # initial values for arrays used in training: 
    X = S
    L = LS
    W = wS
    
    #print(model)
    Bweight = 0
    initial_NB = {}
    preeff_B = {}
    for bkg in Backgrounds:
        xsB=Backgrounds_xsec[(Energy, bkg)] # background cross sections (fb)
        B, LB, wB =  read_ROOT_varfile(Background_files[(Energy, bkg)], idB[bkg], Backgrounds_xsec[(Energy, bkg)])
        preeff_B[bkg] = np.sum(wB)/initial_B[bkg]/Backgrounds_xsec[(Energy, bkg)]
        initial_NB[bkg] =  Luminosity * np.sum(wB)/initial_B[bkg] * bkg_factors # calculate total expected number of events in each background
        Bweight += initial_NB[bkg] # incremenet to total expected number of events
        #print('Background pre-efficiency', bkg, np.sum(wB)/initial_B[bkg]/Backgrounds_xsec[(Energy, bkg)])
        # concatenate lists:
        X = X + B
        L = L + LB
        W = W + wB


    # create testing and training samples:
    #print("Splitting samples into testing and training")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, L, W, test_size=0.99,random_state=seed)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    # in this case signal = 0, backgrounds = i = 1, 2,...
    # (0,0): signal-as-signal -> True positive
    # (i,0): background-as-signal (mis-id) -> False positive
    confmatrix = confusion_matrix(y_test, predictions)
    #print('confusion matrix:')
    #print(confmatrix)
    # signal efficiency:
    total_S = 0
    for j in range(len(Backgrounds)+1):
        total_S += confmatrix[0][j]
    eff_S = confmatrix[0][0]/total_S # signal identified as signal divided by total number of signal events
    # background effiencies:
    eff_B = {}
    for bkg in Backgrounds:
        total_B = 0
        for j in range(len(Backgrounds)+1):
            total_B += confmatrix[idB[bkg]][j]
        eff_B[bkg] = confmatrix[idB[bkg]][0]/total_B
        print(bkg, confmatrix[idB[bkg]][0], total_B)

    #print('Luminosity=', Luminosity)
        
    # initial cross sections into final state:
    #print('Initial signal cross section=', Sweight/Luminosity)
    #print('Initial background cross section=', Bweight/Luminosity)
    #print('-')
    # calculate "significance"
    #print('Initial significance=', Sweight/np.sqrt(Bweight))
    #print('-')
    # print analysis efficiencies
    #print('Signal efficiency=', eff_S)
    #print('Background Efficiencies=', eff_B)
    #print('-')
    #print('Final signal cross section=', Sweight/Luminosity*eff_S)
    # calculate the number of events for the background after the analysis:
    final_NB = {}
    final_NB_total = 0
    for bkg in Backgrounds:
        final_NB[bkg] = initial_NB[bkg] * eff_B[bkg]
        #print('\tNumber of events in', bkg,final_NB[bkg], 'after analysis')
        final_NB_total += final_NB[bkg]
    #print('Final background cross section=', final_NB_total/Luminosity)
    #print('Final significance=', Sweight*eff_S/np.sqrt(final_NB_total))
    #print('-')
    # calculate 95% C.L. limit on expected number of events: 
    S2sigma = np.sqrt(final_NB_total) * 2
    #print('95% C.L. limit on number of signal events=', S2sigma)
    #print('95% C.L. limit on signal cross section in given final state=', S2sigma/Luminosity, 'fb')

    # open files and write all the efficiencies calculated:
    # total efficiency (including what is called "pre-efficiency"
    
    total_eff_S = eff_S*np.sum(wS)/initial_S/xsS
    filestream = open(signal_file.replace('_var.smear' + smeartag + '.root',  smeartag + '.XGBOOST.dat') ,'w')
    filestream.write(str(total_eff_S))
    filestream.close()
    for bkg in Backgrounds:
        filestream = open(Background_files[(Energy, bkg)].replace('_var.smear' + smeartag + '.root',  smeartag + '.XGBOOST.dat') ,'w')
        filestream.write(str(preeff_B[bkg]*eff_B[bkg]))
        filestream.close()




#########################
# Testing starts here   #
#########################

# train the model:
#trained_model, Sweight, Bweight, X_test, y_test = train_xgboost()
# save the model:
#save_model(trained_model, 'trained_model.pkl')
# laod the model: 
#trained_model_test = load_model('trained_model.pkl')
# apply the model:
#apply_xgboost(trained_model_test, Sweight, Bweight, X_test, y_test)
