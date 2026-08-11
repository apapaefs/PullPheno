import ROOT
import sys
import numpy as np
import math
import random
from matplotlib import colors
import matplotlib.gridspec as gridspec # more plotting 
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import time
from time import sleep
import logging
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import fastjet
import cppyy.ll, cppyy

# print root version and check whether to use the new root cppyy treatment:
print('ROOT version:', ROOT.__version__)
newroot = False
if float(ROOT.__version__[:4]) >= 6.32:
    newroot = True
    print('>= 6.32 => Using new cppyy root bindings!')

##########################
# VARIABLES
##########################

# Cuts for jets 
jetPtMin    = 10 # pT in GeV
jetEtaMax   = 5  # HGCAL is up to 3

# cuts for particles to enter jet algorithm:
jcPtMin = 0.1
jcEtaMax = 6.0

# jet algorithm radius parameter
R=0.4

# Cuts for Leptons and Photons
electronPtMin = 10
electronEtaMax = 5

muonPtMin = 10
muonEtaMax = 5

photonPtMin = 10
photonEtaMax = 5

# Debug
debug = False

##########################
# FUNCTIONS
##########################

# choose the next colour -- for plotting
ccount = 0
def next_color():
    global ccount
    colors = ['green', 'orange', 'red', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
    color_chosen = colors[ccount]
    if ccount < 8:
        ccount = ccount + 1
    else:
        ccount = 0    
    return color_chosen

# do not increment colour in this case:
def same_color():
    global ccount
    colors = ['green', 'orange', 'red', 'blue', 'black', 'cyan', 'magenta', 'brown', 'violet'] # 9 colours
    color_chosen = colors[ccount-1]
    return color_chosen

# reset the color counter:
def reset_color():
    global ccount
    ccount = 0

# get momentum of ith particle in format
# E, px, py, pz, id
def ith_momentum(obj, ith):
    if newroot is True:
        return (obj[0][ith],obj[1][ith],obj[2][ith],obj[3][ith],int(obj[4][ith]))
    else:
        return (obj[0*10000+ith],obj[1*10000+ith],obj[2*10000+ith],obj[3*10000+ith],int(obj[4*10000+ith]))

# get all the momenta given the HwSim object
def get_momenta(obj, numprtcl):
    mom = []
    for i in range(0,numprtcl):
        mom.append(ith_momentum(obj, i))
    return np.array(mom, dtype=[('E', 'f8'), ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'), ('id', 'int')])


# calculate the transverse momentum from a particle's list (not pseudojet)
def perp(p):
    pt = math.sqrt( p[1]**2 + p[2]**2 )
    return pt

# calculate the rapidity from a particle's list (not pseudojet)
def rapidity(p):
    rapd = 0.5 * math.log( (p[0] + p[3]) / (p[0] - p[3]) )
    return rapd

# calculate the pseudorapidity from a particle's list (not pseudojet)
def pseudorapidity(p):
    pt = perp(p)
    theta = np.arctan(pt/p[3])
    if theta < 0:
       theta += np.pi
    return -np.log(np.tan(theta/2));

# calculate the phi of a particle from list:
def phi(p):
    if p[1] == 0 and p[2] == 0:
        return 0
    else:
        return np.arctan2(p[2],p[1])


# convert a pseudojet to a list
def convert_to_array(pseudojet):
    return [pseudojet.e, pseudojet.px, pseudojet.py, pseudojet.pz]

# add four momenta coming from lists
def add_4momenta(p1, p2):
    summedvec = []
    for (item1, item2) in zip(p1, p2):
        summedvec.append(item1 + item2)
    return summedvec

# calculate the invariant mass of a particle's list (not pseudojet)
def invmass(fourvector):
    return math.sqrt(fourvector[0]**2 - fourvector[1]**2 - fourvector[2]**2 - fourvector[3]**2)

# plot histogram of a single observable contained in DATA given by plotname
def histogram(DATA, plotname, xlabel, nbins=50):
    print('---')
    print('plotting')

    # plot settings ########
    plot_type = plotname # the name of the plot
    # the following labels are in LaTeX, but instead of a single slash, two "\\" are required.
    ylab = 'fraction/bin' # the ylabel
    xlab = xlabel # the x label
    # log scale?
    ylog = False # whether to plot y in log scale
    xlog = False # whether to plot x in log scale

    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(False)

    #print('len(DATA)=', len(DATA))
    bins, edges = np.histogram(np.array(DATA), bins=nbins)
    #print(bins)
    errors = np.divide(np.sqrt(bins), bins, out=np.zeros_like(np.sqrt(bins)), where=bins!=0.)
    #print errors
    bins = bins/float(len(DATA))
    errors = bins*errors
    print(bins)
    print(errors)
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    Y = np.array([bins,bins]).T.flatten()
    #print('X=',X)
    #print('Y=',Y)
    plt.plot(X,Y, label='', color='red', lw=1)
    center = (edges[:-1] + edges[1:]) / 2
    plt.errorbar(center, bins, yerr=errors, color='red', lw=0, elinewidth=1, capsize=1)
    

    # set the ticks, labels and limits etc.
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_xlabel(xlab, fontsize=20)
    
    # choose x and y log scales
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    # set the limits on the x and y axes if required below:
    #xmin = 0.
    #xmax = 1500.
    #ymin = 0.
    #ymax = 0.09
    #plt.xlim([0,180])
    #plt.ylim([0.08,0.12])

    # create legend and plot/font size
    #ax.legend()
    #ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})

    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    infile = plot_type + '.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)


# plot histogram of a single observable given by plotname: DATA_array contains results from several root files and plot_type defines the plot type
def histogram_multi(DATA_array, plot_type, plotnames_multi, xlabel, nbins=50, xmin=0, xmax=180, ymin=0.06, ymax=0.15):
    print('---')
    print('plotting')

    # plot settings ########
    # the following labels are in LaTeX, but instead of a single slash, two "\\" are required.
    ylab = 'fraction/bin' # the ylabel
    xlab = xlabel # the x label
    # log scale?
    ylog = False # whether to plot y in log scale
    xlog = False # whether to plot x in log scale

    # construct the axes for the plot
    # no need to modify this if you just need one plot
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(False)

    #print('len(DATA)=', len(DATA))
    dd = 0
    for DATA in DATA_array:
        bins, edges = np.histogram(np.array(DATA), bins=nbins)
        #print(bins)
        errors = np.divide(np.sqrt(bins), bins, out=np.zeros_like(np.sqrt(bins)), where=bins!=0.)
        #print errors
        bins = bins/float(len(DATA))
        errors = bins*errors
        print(bins)
        print(errors)
        left,right = edges[:-1],edges[1:]
        X = np.array([left,right]).T.flatten()
        Y = np.array([bins,bins]).T.flatten()
        #print('X=',X)
        #print('Y=',Y)
        plt.plot(X,Y, label=plotnames_multi[dd], color=next_color(), lw=1)
        center = (edges[:-1] + edges[1:]) / 2
        plt.errorbar(center, bins, yerr=errors, color=same_color(), lw=0, elinewidth=1, capsize=1)
        dd = dd+1
    

    # set the ticks, labels and limits etc.
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_xlabel(xlab, fontsize=20)
    
    # choose x and y log scales
    if ylog:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    if xlog:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    # set the limits on the x and y axes if required below:
    #xmin = 0.
    #xmax = 1500.
    #ymin = 0.
    #ymax = 0.09
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])

    # create legend and plot/font size
    ax.legend()
    ax.legend(loc="upper right", numpoints=1, frameon=False, prop={'size':8})

    # save the figure
    print('saving the figure')
    # save the figure in PDF format
    infile = plot_type + '.dat'
    print('output in', infile.replace('.dat','.pdf'))
    plt.savefig(infile.replace('.dat','.pdf'), bbox_inches='tight')
    plt.close(fig)
    reset_color()

# grab the events and their matching weights from the ROOT file
def get_events(treein, filename, maxevents=1000000):
    if maxevents > treein.GetEntries():
        maxevents = treein.GetEntries()
    print('Getting', maxevents, 'events from', filename)
    events = []
    event_weights = []
    for entryNum in tqdm(range(0,maxevents)):
        # get the entry from the tree
        treein.GetEntry(entryNum)
        event_weights.append(float(getattr(treein, "evweight")))
        # get the number of particles in the event
        # and the objects array
        numevents = getattr(treein,"numparticles")
        #print(numevents)
        objects = getattr(treein,"objects")
        if newroot is True:
            objects = np.asarray(objects).reshape((8, 10000))        
        # convert the objects array to the right format
        momenta = get_momenta(objects, numevents)
        events.append(momenta)
    return events, event_weights

# convert to fastjet, but only if the pt is > minptc
def convert_tofj(momin, minptc, maxrapc):
    arrayout = []
    for mm in range(len(momin)):
        #print(momin[mm][1], momin[mm][2], momin[mm][3], momin[mm][0])
        fj = fastjet.PseudoJet(momin[mm][1], momin[mm][2], momin[mm][3], momin[mm][0])
        if fj.perp() > minptc and abs(fj.eta()) < maxrapc:
            arrayout.append(fj)
        fj.set_user_index(int(momin[mm][4]))
    return arrayout

def get_reconstructed(events, filename, jetalgo, jetR, maxevents=100000):
    if maxevents > len(events):
        maxevents = len(events)
    print('Analyzing', maxevents, 'events from', filename)
    # put the Higgs momenta into an array:
    higgs = []
    # return the cluster:
    clusters_jets = []
    # return the unclustered objects:
    electrons_reco = []
    positrons_reco = []
    muons_reco = []
    antimuons_reco = []
    photons_reco = []
    # jet algorithm
    jetdef = fastjet.JetDefinition(jetalgo, jetR)
    # loop over events and analyze:
    for yy in tqdm(range(0,maxevents)):
        # put the momenta for clustering into array:
        momtocluster = []
        # and the rest into other arrays:
        electrons = []
        positrons = []
        muons = []
        antimuons = []
        photons = []
        # all the momenta from this event
        momenta = events[yy]
        #print(momenta)jcEtaMax
        for mm in range(0,len(momenta)):
            pt = perp(momenta[mm])
            eta = pseudorapidity(momenta[mm])
            ph = phi(momenta[mm])
            #print(pt, eta, ph)
            if abs(momenta[mm][4]) == 11 and pt > electronPtMin and abs(eta) < electronEtaMax:
                if momenta[mm][4] > 0:
                    electrons.append(momenta[mm])
                else:
                    positrons.append(momenta[mm])
            elif abs(momenta[mm][4]) == 13 and pt > muonPtMin and abs(eta) < muonEtaMax:
                if momenta[mm][4] > 0:
                    muons.append(momenta[mm])
                else:
                    antimuons.append(momenta[mm])
            elif momenta[mm][4] == 22 and pt > photonPtMin and abs(eta) < photonEtaMax:
                 photons.append(momenta[mm])
            else: # if not electron/muon (or anti) or muon, put into jet clustering 
                momtocluster.append(momenta[mm])

        # convert to fastjet:
        momfj_electrons = convert_tofj(electrons, electronPtMin, electronEtaMax)
        momfj_positrons = convert_tofj(positrons, electronPtMin, electronEtaMax)
        momfj_muons = convert_tofj(muons, muonPtMin, muonEtaMax)
        momfj_antimuons = convert_tofj(antimuons, electronPtMin,electronEtaMax)
        momfj_photons = convert_tofj(photons, photonPtMin, photonEtaMax)
        electrons_reco.append(momfj_electrons)
        positrons_reco.append(momfj_positrons)
        muons_reco.append(momfj_muons)
        antimuons_reco.append(momfj_antimuons)
        photons_reco.append(momfj_photons)
        # get the jets and push to bigger array:
        momfj = convert_tofj(momtocluster, jcPtMin,jcEtaMax) # convert to fastjet
        cluster = fastjet.ClusterSequence(momfj, jetdef)
        clusters_jets.append(cluster)
        
    return clusters_jets, electrons_reco, positrons_reco, muons_reco, antimuons_reco, photons_reco


# from https://gitlab.cern.ch/cms-sw/cmssw/blob/e303d9f2c3d4f25397db5feb7ad59d2f20c842f2/PhysicsTools/HeppyCore/python/utils/deltar.py
def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = p1 - p2
    while res > np.pi:
        res -= 2*np.pi
    while res < -np.pi:
        res += 2*np.pi
    return res

##########################
# MAIN ANALYSIS FUNCTION #
##########################
def analyze(clusters_jets, electrons, positrons, muons, antimuons, photons):
    # define dictionary with data to return at the end of the analysis
    DATA = {}
    # example variables:
    DATA['ptleptons'] = []
    DATA['ptjets'] = []
    passed = 0
    # loop over events
    for ee in tqdm(range(0,len(clusters_jets))):
        # sort the jets, with jet pt minimum jetPtMin
        events_jets = fastjet.sorted_by_pt(clusters_jets[ee].inclusive_jets(jetPtMin))
        for jet in events_jets:
            if abs(jet.eta()) < jetEtaMax:
                DATA['ptjets'].append(jet.perp())
        for electron in electrons[ee]:
            DATA['ptleptons'].append(electron.perp())
        for muon in muons[ee]:
            DATA['ptleptons'].append(muon.perp())
        for positron in positrons[ee]:
            DATA['ptleptons'].append(electron.perp())
        for antimuon in antimuons[ee]:
            DATA['ptleptons'].append(antimuon.perp())
        passed = passed + 1
    print('Passed cuts fraction =', passed/len(clusters_jets))
    return DATA
    #histogram(relpull_array, 'relpull_' + plotname, 'Relative pull angle $\\theta_{12}$ (degrees)', nbins=10)

###########################
# MAIN PROGRAM
###########################

if len(sys.argv) < 2:
    print("USAGE: %s <input ROOT file(s)>"%(sys.argv[0]))
    sys.exit(1)

inFileName = sys.argv[1]
plotnames_multi = []
plotnames_multi.append(inFileName)

# read the ROOT file and get the data
inFile = ROOT.TFile.Open(inFileName ,"READ")
tree = inFile.Get("Data")

# analyze
print("Analyzing", inFileName)
print(inFileName, "contains:", tree.GetEntries(), "events")

# maximum number of events to analyze:
Nmax = 100000

# get all the event momenta from the root file:
events, event_weights = get_events(tree, inFileName, maxevents=Nmax)
print("Sum of event weights:", sum(event_weights))

# convert to jets
# in this case we ignore the (stable) Higgs boson and neutrinos
cluster_jets, electrons, positrons, muons, antimuons, photons =  get_reconstructed(events, inFileName, fastjet.antikt_algorithm, R, maxevents=Nmax)


# analyze the jets in the events
OUTPUT = analyze(cluster_jets, electrons, positrons, muons, antimuons, photons)

# histogram: 
histogram(OUTPUT['ptleptons'], 'ptleptons', '$p_T$ of leptons [GeV]', nbins=10)
histogram(OUTPUT['ptjets'], 'ptjets', '$p_T$ of jets [GeV]', nbins=10)
