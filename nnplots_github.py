#!/bin/env python
import numpy
numpy.random.seed(1337)
import pandas
import ROOT
import random
import time
import math
import sys
import csv
import os

nnoutput_name = "higgs_dy"   # name of outputs of the neural network
nnoutput_path = "nnoutput"           # path to output of neural network
svfitoutput_name = "ditau_mass_svfit"      # name of outputs of SVfit   
svfitoutput_path = "svfitoutput"     # path to output of SVfit
inputvalues_path = "nninput"         # path to input values
output_name = "higgs_dy"     # name of plots
decaymode = "no" # choose "fulllep","semilep" or "fullhad" to get plots with only events with one decay channel
bias_correction = "no"   # choose "gen" or "reco" to correct the bias using the gen mass or the reconstructed mass
signal = "110GeV" # choose mass of Higgs samples to compare with the DY background
signalrange = "no" # choose "yes" to get signal to background ratio in the signalrange
signalrange_tight = "no" # choose "yes" to get signal to background ratio in a tight range around the signal
epochs = 400


####################   getting the neural network outputs ###########################
nn_output_name = "%s/nnoutput_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_100GeV_name = "%s/nnoutput_100GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_110GeV_name = "%s/nnoutput_110GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_125GeV_name = "%s/nnoutput_125GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_140GeV_name = "%s/nnoutput_140GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_180GeV_name = "%s/nnoutput_180GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_250GeV_name = "%s/nnoutput_250GeV_%s.csv" % (nnoutput_path,nnoutput_name)
nn_output_dy_name = "%s/nnoutput_dy_%s.csv" % (nnoutput_path,nnoutput_name)
loss_values_name = "%s/nnoutput_loss_%s.csv" % (nnoutput_path,nnoutput_name)
val_loss_values_name = "%s/nnoutput_val_loss_%s.csv" % (nnoutput_path,nnoutput_name)

ditaumass_nn = pandas.read_csv(nn_output_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_100GeV = pandas.read_csv(nn_output_100GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_110GeV = pandas.read_csv(nn_output_110GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_125GeV = pandas.read_csv(nn_output_125GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_140GeV = pandas.read_csv(nn_output_140GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_180GeV = pandas.read_csv(nn_output_180GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_250GeV = pandas.read_csv(nn_output_250GeV_name, delim_whitespace=False,header=None).values[:,0]
ditaumass_nn_dy = pandas.read_csv(nn_output_dy_name, delim_whitespace=False,header=None).values[:,0]
loss_values = pandas.read_csv(loss_values_name, delim_whitespace=False,header=None).values[:,0]
val_loss_values = pandas.read_csv(val_loss_values_name, delim_whitespace=False,header=None).values[:,0]

###############  getting test values
test_input_name = "%s/nninput_test_nostand_small.csv" % inputvalues_path
test_ditaumass_name = "%s/nntarget_test_nostand_small.csv" % inputvalues_path
ditauvismass_name = "%s/ditauvismass_test_nostand_small.csv" % inputvalues_path
collinear_ditaumass_name = "%s/ditaucollinearmass_test_nostand_small.csv" % inputvalues_path
test_ditaumass_100GeV_name = "%s/nntarget_test_100GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_110GeV_name = "%s/nntarget_test_110GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_125GeV_name = "%s/nntarget_test_125GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_140GeV_name = "%s/nntarget_test_140GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_180GeV_name = "%s/nntarget_test_180GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_250GeV_name = "%s/nntarget_test_250GeV_nostand_small.csv" % inputvalues_path
test_ditaumass_dy_name = "%s/nntarget_test_dy_nostand_small.csv" % inputvalues_path

test_input_selected = pandas.read_csv(test_input_name, delim_whitespace=False,header=None).values[:,:]
test_ditaumass_selected = pandas.read_csv(test_ditaumass_name, delim_whitespace=False,header=None).values[:,0]
test_ditauvismass_selected = pandas.read_csv(ditauvismass_name, delim_whitespace=False,header=None).values[:,0]
test_ditaucollinearmass_selected = pandas.read_csv(collinear_ditaumass_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_100GeV = pandas.read_csv(test_ditaumass_100GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_110GeV = pandas.read_csv(test_ditaumass_110GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_125GeV = pandas.read_csv(test_ditaumass_125GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_140GeV = pandas.read_csv(test_ditaumass_140GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_180GeV = pandas.read_csv(test_ditaumass_180GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_250GeV = pandas.read_csv(test_ditaumass_250GeV_name, delim_whitespace=False,header=None).values[:,0]
test_ditaumass_dy = pandas.read_csv(test_ditaumass_dy_name, delim_whitespace=False,header=None).values[:,0]

#####################  getting the SVfit output ###############################################
ditaumass_svfit =  pandas.read_csv("%s/%s_wholerange.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit_gen =  pandas.read_csv("%s/%s_wholerange_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit_decaymode =  pandas.read_csv("%s/%s_wholerange_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit100GeV =  pandas.read_csv("%s/%s_100GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit100GeV_gen =  pandas.read_csv("%s/%s_100GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit100GeV_decaymode =  pandas.read_csv("%s/%s_100GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit110GeV =  pandas.read_csv("%s/%s_110GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit110GeV_gen =  pandas.read_csv("%s/%s_110GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit110GeV_decaymode =  pandas.read_csv("%s/%s_110GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit125GeV = pandas.read_csv("%s/%s_125GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit125GeV_gen = pandas.read_csv("%s/%s_125GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit125GeV_decaymode = pandas.read_csv("%s/%s_125GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfitdy =  pandas.read_csv("%s/%s_dy.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfitdy_gen =  pandas.read_csv("%s/%s_dy_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfitdy_decaymode =  pandas.read_csv("%s/%s_dy_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit140GeV = pandas.read_csv("%s/%s_140GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit140GeV_gen = pandas.read_csv("%s/%s_140GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit140GeV_decaymode = pandas.read_csv("%s/%s_140GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit180GeV = pandas.read_csv("%s/%s_180GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit180GeV_gen = pandas.read_csv("%s/%s_180GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit180GeV_decaymode = pandas.read_csv("%s/%s_180GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit250GeV = pandas.read_csv("%s/%s_250GeV.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit250GeV_gen = pandas.read_csv("%s/%s_250GeV_gen.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
ditaumass_svfit250GeV_decaymode = pandas.read_csv("%s/%s_250GeV_decaymode.csv" % (svfitoutput_path,svfitoutput_name), delim_whitespace=False,header=None).values[:,0]
#################### nn values for different decaymodes ######################
ditaumass_nn_decaymode = []
if decaymode == "fulllep":
    for g,ditaumass_value in enumerate(ditaumass_nn):
        if test_input_selected[g,0] == 1.0:
            ditaumass_nn_decaymode.append(ditaumass_value)
    ditaumass_nn = ditaumass_nn_decaymode
if decaymode == "semilep":
    for g,ditaumass_value in enumerate(ditaumass_nn):
        if test_input_selected[g,1] == 1.0 or test_input_selected[g,2] == 1.0:
            ditaumass_nn_decaymode.append(ditaumass_value)
    ditaumass_nn = ditaumass_nn_decaymode
if decaymode == "fullhad":
    for g,ditaumass_value in enumerate(ditaumass_nn):
        if test_input_selected[g,3] == 1.0:
            ditaumass_nn_decaymode.append(ditaumass_value)
    ditaumass_nn = ditaumass_nn_decaymode

#################### test values for different decaymodes ######################
test_ditaumass_selected_decaymode = []
if decaymode == "fulllep":
    for g,ditaumass_value in enumerate(test_ditaumass_selected):
        if test_input_selected[g,0] == 1.0:
            test_ditaumass_selected_decaymode.append(ditaumass_value)
    test_ditaumass_selected = test_ditaumass_selected_decaymode
if decaymode == "semilep":
    for g,ditaumass_value in enumerate(test_ditaumass_selected):
        if test_input_selected[g,1] == 1.0 or test_input_selected[g,2] == 1.0:
            test_ditaumass_selected_decaymode.append(ditaumass_value)
    test_ditaumass_selected = test_ditaumass_selected_decaymode
if decaymode == "fullhad":
    for g, ditaumass_value in enumerate(test_ditaumass_selected):
        if test_input_selected[g,3] == 1.0:
            test_ditaumass_selected_decaymode.append(ditaumass_value)
    test_ditaumass_selected = test_ditaumass_selected_decaymode


#################### svfit values for different decaymodes #####################
ditaumass_svfit_decaymode_loop = []
ditaumass_svfit_gen_decaymode_loop = []
if decaymode == "fulllep":
    for g,ditaumass_value in enumerate(ditaumass_svfit):
        if ditaumass_svfit_decaymode[g] == 3.0:
            ditaumass_svfit_decaymode_loop.append(ditaumass_value)
            ditaumass_svfit_gen_decaymode_loop.append(ditaumass_svfit_gen[g])
    ditaumass_svfit = ditaumass_svfit_decaymode_loop
    ditaumass_svfit_gen = ditaumass_svfit_gen_decaymode_loop
if decaymode == "semilep":
    for g,ditaumass_value in enumerate(ditaumass_svfit):
        if ditaumass_svfit_decaymode[g] == 4.0:
            ditaumass_svfit_decaymode_loop.append(ditaumass_value)
            ditaumass_svfit_gen_decaymode_loop.append(ditaumass_svfit_gen[g])
    ditaumass_svfit = ditaumass_svfit_decaymode_loop
    ditaumass_svfit_gen = ditaumass_svfit_gen_decaymode_loop
if decaymode == "fullhad":
    for g,ditaumass_value in enumerate(ditaumass_svfit):
        if ditaumass_svfit_decaymode[g] == 5.0:
            ditaumass_svfit_decaymode_loop.append(ditaumass_value)
            ditaumass_svfit_gen_decaymode_loop.append(ditaumass_svfit_gen[g])
    ditaumass_svfit = ditaumass_svfit_decaymode_loop
    ditaumass_svfit_gen = ditaumass_svfit_gen_decaymode_loop

############## signal range to compare Higgs signal with DY background #################3
if signalrange == "yes":
    if signalrange_tight == "yes":
        output_file_name = "%s_%s_tight.txt" % (output_name,signal)
    elif signalrange_tight == "no":
        output_file_name = "%s_%s.txt" % (output_name,signal)
    output_file = open(output_file_name,'w')
    sys.stdout = output_file

if signal =="100GeV":
    signal_tight_left = 95
    signal_tight_right = 105
    signal_left = 90
    signal_right = 110
    signal_bkg_left = 80
    signal_bkg_right = 110
    if signalrange == "yes":
        signal_bkg_left = signal_left
        signal_bkg_right = signal_right
    if signalrange_tight == "yes":
        signal_left = signal_tight_left
        signal_right = signal_tight_right

if signal =="110GeV":
    signal_tight_left = 105
    signal_tight_right = 115
    signal_left = 100
    signal_right = 120
    signal_bkg_left = 80
    signal_bkg_right = 120
    if signalrange == "yes":
        signal_bkg_left = signal_left
        signal_bkg_right = signal_right
    if signalrange_tight == "yes":
        signal_left = signal_tight_left
        signal_right = signal_tight_right

if signal =="125GeV":
    signal_tight_left = 120
    signal_tight_right = 130
    signal_left = 115
    signal_right = 135
    signal_bkg_left = 80
    signal_bkg_right = 135
    if signalrange == "yes":
        signal_bkg_left = signal_left
        signal_bkg_right = signal_right
    if signalrange_tight == "yes":
        signal_left = signal_tight_left
        signal_right = signal_tight_right

if signal =="140GeV":
    signal_tight_left = 135
    signal_tight_right = 145
    signal_left = 130
    signal_right = 150
    signal_bkg_left = 80
    signal_bkg_right = 150
    if signalrange == "yes":
        signal_bkg_left = signal_left
        signal_bkg_right = signal_right
    if signalrange_tight == "yes":
        signal_left = signal_tight_left
        signal_right = signal_tight_right



#############    preparing the histograms       ##############################
ROOT.TGaxis.SetMaxDigits(3)
#histogram of ditau mass using neural network and SVfit
histtitle = "reconstruct di-#tau mass using a neural network and SVfit"
histditaumass = ROOT.TH1D("ditaumass",histtitle,60,50,350)
histditaumass.SetTitleSize(0.3,"t")
histditaumass.GetXaxis().SetTitle("")
histditaumass.GetXaxis().SetLabelSize(0)
histditaumass.GetYaxis().SetTitle("number of occurence")
histditaumass.GetYaxis().SetTitleSize(0.06)
histditaumass.GetYaxis().SetTitleOffset(0.75)
histditaumass.GetYaxis().SetLabelSize(0.06)
histditaumass.SetLineColor(2)
histditaumass.SetLineWidth(3)
histditaumass.SetStats(0)
histditaumassgen = ROOT.TH1D("ditaumassgen","di-#tau_{gen} mass and visible mass",60,50,350)
histditaumassgen.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassgen.GetYaxis().SetTitle("number of occurence")
histditaumassgen.GetXaxis().SetTitleSize(0.04)
histditaumassgen.GetXaxis().SetLabelSize(0.04)
histditaumassgen.GetXaxis().SetLabelOffset(0.01)
histditaumassgen.GetXaxis().SetTitleOffset(1.15)
histditaumassgen.GetYaxis().SetTitleSize(0.04)
histditaumassgen.GetYaxis().SetLabelSize(0.04)
histditaumassgen.GetYaxis().SetLabelOffset(0.01)
histditaumassgen.SetLineColor(2)
histditaumassgen.SetLineWidth(3)
histditaumassgen.SetStats(0)
histditauvismass = ROOT.TH1D("ditauvismass","reconstructed di-#tau vismass using neural network",60,50,350)
histditauvismass.SetLineColor(6)
histditauvismass.SetLineWidth(3)
histditauvismass.SetLineStyle(7)
histditauvismass.SetStats(0)
histditaumassnn = ROOT.TH1D("ditaumassnn","reconstructed di-#tau mass using neural network",60,50,350)
histditaumassnn.SetLineColor(4)
histditaumassnn.SetLineWidth(3)
histditaumassnn.SetLineStyle(7)
histditaumassnn.SetStats(0)
histditaumasssvfit = ROOT.TH1D("ditaumasssvfit","di-#tau mass using SVfit",60,50,350)
histditaumasssvfit.SetLineColor(8)
histditaumasssvfit.SetLineWidth(3)
histditaumasssvfit.SetLineStyle(2)
histditaumasssvfit.SetStats(0)
histditaucollmass = ROOT.TH1D("ditaucollmass","collinear ditau mass",60,50,350)
histditaucollmass.SetLineColor(7)
histditaucollmass.SetLineWidth(3)
histditaucollmass.SetLineStyle(3)
histditaucollmass.SetStats(0)


histditaumassnncorr = ROOT.TH2D("ditaumassnncorr","di-#tau_{gen} mass vs di-#tau mass",350,50,400,350,50,400)
histditaumassnncorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumassnncorr.GetXaxis().SetTitleSize(0.04)
histditaumassnncorr.GetXaxis().SetLabelSize(0.04)
histditaumassnncorr.GetXaxis().SetTitleOffset(1.15)
histditaumassnncorr.GetXaxis().SetLabelOffset(0.01)
histditaumassnncorr.GetYaxis().SetTitleSize(0.04)
histditaumassnncorr.GetYaxis().SetLabelSize(0.04)
histditaumassnncorr.GetYaxis().SetLabelOffset(0.01)
histditaumassnncorr.GetYaxis().SetTitleOffset(1.2)
histditaumassnncorr.SetStats(0)
histditaumasssvfitcorr = ROOT.TH2D("ditaumasssvfitcorr","di-#tau_{gen} mass vs di-#tau mass",350,50,400,350,50,400)
histditaumasssvfitcorr.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorr.GetYaxis().SetTitle("di-#tau mass [GeV]")
histditaumasssvfitcorr.GetXaxis().SetTitleSize(0.04)
histditaumasssvfitcorr.GetXaxis().SetLabelSize(0.04)
histditaumasssvfitcorr.GetXaxis().SetTitleOffset(1.15)
histditaumasssvfitcorr.GetXaxis().SetLabelOffset(0.01)
histditaumasssvfitcorr.GetYaxis().SetTitleSize(0.04)
histditaumasssvfitcorr.GetYaxis().SetLabelSize(0.04)
histditaumasssvfitcorr.GetYaxis().SetLabelOffset(0.01)
histditaumasssvfitcorr.GetYaxis().SetTitleOffset(1.2)
histditaumasssvfitcorr.SetStats(0)

profditaumassnncorrrms = ROOT.TProfile("profditaumassnncorrrms","di-#tau_{gen} mass vs mean(di-#tau mass)",300,50,350,"s")

histditaumassnnrms =ROOT.TH1D("ditaumassnnrms","RMS per di-#tau_{gen} mass",300,50,350)
histditaumassnnrms.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnrms.GetYaxis().SetTitle("RMS")
histditaumassnnrms.GetXaxis().SetTitleSize(0.04)
histditaumassnnrms.GetXaxis().SetLabelSize(0.04)
histditaumassnnrms.GetXaxis().SetTitleOffset(1.15)
histditaumassnnrms.GetXaxis().SetLabelOffset(0.01)
histditaumassnnrms.GetYaxis().SetTitleSize(0.04)
histditaumassnnrms.GetYaxis().SetLabelSize(0.04)
histditaumassnnrms.GetYaxis().SetLabelOffset(0.01)
histditaumassnnrms.SetStats(0)
histditaumassnnrms.SetMarkerStyle(7)

histditaumassnnres = ROOT.TH1D("resolution","relative difference per event using neural network",80,-1,1)
histditaumassnnres.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres.GetYaxis().SetTitle("number of occurence")
histditaumassnnres.GetXaxis().SetTitleSize(0.04)
histditaumassnnres.GetXaxis().SetLabelSize(0.04)
histditaumassnnres.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres.GetYaxis().SetTitleSize(0.04)
histditaumassnnres.GetYaxis().SetLabelSize(0.04)
histditaumassnnres.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres.SetLineWidth(3)
histditaumassnnrescomp = ROOT.TH1D("NN","relative difference per event",80,-1,1)
histditaumassnnrescomp.GetXaxis().SetTitle("relative difference per event")
histditaumassnnrescomp.GetYaxis().SetTitle("number of occurence")
histditaumassnnrescomp.GetXaxis().SetTitleSize(0.04)
histditaumassnnrescomp.GetXaxis().SetLabelSize(0.04)
histditaumassnnrescomp.GetXaxis().SetTitleOffset(1.15)
histditaumassnnrescomp.GetXaxis().SetLabelOffset(0.01)
histditaumassnnrescomp.GetYaxis().SetTitleSize(0.04)
histditaumassnnrescomp.GetYaxis().SetLabelSize(0.04)
histditaumassnnrescomp.GetYaxis().SetLabelOffset(0.01)
histditaumassnnrescomp.GetYaxis().SetTitleOffset(1.2)
histditaumassnnrescomp.SetLineColor(4)
histditaumassnnrescomp.SetLineWidth(3)
histditaumasssvfitres = ROOT.TH1D("SVfit","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres.SetLineColor(8)
histditaumasssvfitres.SetLineWidth(3)

histditaumassnncorrres = ROOT.TH2D("ditaumassnncorrres","relative difference per event",300,50,350,80,-1,1)
histditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnncorrres.GetYaxis().SetTitle("relative difference per event")
histditaumassnncorrres.GetXaxis().SetTitleSize(0.04)
histditaumassnncorrres.GetXaxis().SetLabelSize(0.04)
histditaumassnncorrres.GetXaxis().SetTitleOffset(1.15)
histditaumassnncorrres.GetXaxis().SetLabelOffset(0.01)
histditaumassnncorrres.GetYaxis().SetTitleSize(0.04)
histditaumassnncorrres.GetYaxis().SetLabelSize(0.04)
histditaumassnncorrres.GetYaxis().SetLabelOffset(0.01)
histditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
histditaumassnncorrres.SetStats(0)
histditaumasssvfitcorrres = ROOT.TH2D("ditaumasssvfitcorrres","relative difference per event",300,50,350,80,-1,1)
histditaumasssvfitcorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumasssvfitcorrres.GetYaxis().SetTitle("relative difference per event")
histditaumasssvfitcorrres.GetXaxis().SetTitleSize(0.04)
histditaumasssvfitcorrres.GetXaxis().SetLabelSize(0.04)
histditaumasssvfitcorrres.GetXaxis().SetTitleOffset(1.15)
histditaumasssvfitcorrres.GetXaxis().SetLabelOffset(0.01)
histditaumasssvfitcorrres.GetYaxis().SetTitleSize(0.04)
histditaumasssvfitcorrres.GetYaxis().SetLabelSize(0.04)
histditaumasssvfitcorrres.GetYaxis().SetLabelOffset(0.01)
histditaumasssvfitcorrres.GetYaxis().SetTitleOffset(1.2)
histditaumasssvfitcorrres.SetStats(0)

bin_number = 50
before_limit = 100
if bias_correction == "gen":
    bin_number = 300
    before_limit = 350
if bias_correction == "reco":
    bin_number = 400
    before_limit = 450
profditaumassnncorrresbefore = ROOT.TProfile("profditaumassnncorrresbefore","bias in the di-#tau mass reconstruction",bin_number,50,before_limit)
if bias_correction == "gen":
    profditaumassnncorrresbefore.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
if bias_correction == "reco":
    profditaumassnncorrresbefore.GetXaxis().SetTitle("di-#tau mass [GeV]")
profditaumassnncorrresbefore.GetYaxis().SetTitle("bias")
profditaumassnncorrresbefore.GetXaxis().SetTitleSize(0.04)
profditaumassnncorrresbefore.GetXaxis().SetLabelSize(0.04)
profditaumassnncorrresbefore.GetXaxis().SetTitleOffset(1.15)
profditaumassnncorrresbefore.GetXaxis().SetLabelOffset(0.01)
profditaumassnncorrresbefore.GetYaxis().SetTitleSize(0.04)
profditaumassnncorrresbefore.GetYaxis().SetLabelSize(0.04)
profditaumassnncorrresbefore.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrresbefore.GetYaxis().SetLabelOffset(0.01)
profditaumassnncorrresbefore.SetStats(0)
profditaumassnncorrresbefore.SetLineColor(4)
profditaumassnncorrresbefore.SetMarkerStyle(7)
profditaumassnncorrresbefore.SetMarkerColor(4)
profditaumassnncorrres = ROOT.TProfile("profditaumassnncorrres","bias in the di-#tau mass reconstruction",300,50,350)
profditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrres.GetYaxis().SetTitle("bias")
profditaumassnncorrres.GetXaxis().SetTitleSize(0.04)
profditaumassnncorrres.GetXaxis().SetLabelSize(0.04)
profditaumassnncorrres.GetXaxis().SetTitleOffset(1.15)
profditaumassnncorrres.GetXaxis().SetLabelOffset(0.01)
profditaumassnncorrres.GetYaxis().SetTitleSize(0.04)
profditaumassnncorrres.GetYaxis().SetLabelSize(0.04)
profditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrres.GetYaxis().SetLabelOffset(0.01)
profditaumassnncorrres.SetStats(0)
profditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres = ROOT.TH1D("histprofditaumassnncorrres","bias in the di-#tau mass reconstruction",300,50,350)
histprofditaumassnncorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
#histprofditaumassnncorrres.GetXaxis().SetTitle("di-#tau mass [GeV]")
histprofditaumassnncorrres.GetYaxis().SetTitle("bias")
histprofditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrres.GetXaxis().SetTitleSize(0.04)
histprofditaumassnncorrres.GetXaxis().SetLabelSize(0.04)
histprofditaumassnncorrres.GetXaxis().SetTitleOffset(1.15)
histprofditaumassnncorrres.GetXaxis().SetLabelOffset(0.01)
histprofditaumassnncorrres.GetYaxis().SetTitleSize(0.04)
histprofditaumassnncorrres.GetYaxis().SetLabelSize(0.04)
histprofditaumassnncorrres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrres.GetYaxis().SetLabelOffset(0.01)
histprofditaumassnncorrres.SetStats(0)
histprofditaumassnncorrres.SetLineColor(4)
histprofditaumassnncorrres.SetMarkerStyle(7)
histprofditaumassnncorrres.SetMarkerColor(4)
profditaumasssvfitcorrresbefore = ROOT.TProfile("profditaumasssvfitcorrresbefore","bias in the di-#tau mass reconstruction",bin_number,50,before_limit)
profditaumasssvfitcorrresbefore.SetStats(0)
profditaumasssvfitcorrresbefore.SetLineColor(8)
profditaumasssvfitcorrresbefore.SetMarkerStyle(7)
profditaumasssvfitcorrresbefore.SetMarkerColor(8)
profditaumasssvfitcorrres = ROOT.TProfile("profditaumasssvfitcorrres","bias in the di-#tau mass reconstruction",300,50,350)
profditaumasssvfitcorrres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumasssvfitcorrres.GetYaxis().SetTitle("bias")
profditaumasssvfitcorrres.GetXaxis().SetTitleSize(0.04)
profditaumasssvfitcorrres.GetXaxis().SetLabelSize(0.04)
profditaumasssvfitcorrres.GetXaxis().SetTitleOffset(1.15)
profditaumasssvfitcorrres.GetXaxis().SetLabelOffset(0.01)
profditaumasssvfitcorrres.GetYaxis().SetTitleSize(0.04)
profditaumasssvfitcorrres.GetYaxis().SetLabelSize(0.04)
profditaumasssvfitcorrres.GetYaxis().SetTitleOffset(1.2)
profditaumasssvfitcorrres.GetYaxis().SetLabelOffset(0.01)
profditaumasssvfitcorrres.SetStats(0)
profditaumasssvfitcorrres.SetMarkerStyle(7)
histprofditaumasssvfitcorrres = ROOT.TH1D("histprofditaumasssvfitcorrres","mean of resolution per di-#tau_{gen} mass",300,50,350)
histprofditaumasssvfitcorrres.SetStats(0)
histprofditaumasssvfitcorrres.SetLineColor(8)
histprofditaumasssvfitcorrres.SetMarkerStyle(7)
histprofditaumasssvfitcorrres.SetMarkerColor(8)

profditaumassnncorrabsres = ROOT.TProfile("profditaumassnncorrabsres","average of the absolute relative differences per event",300,50,350)
profditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumassnncorrabsres.GetYaxis().SetTitle("average of the absolute relative difference per event")
profditaumassnncorrabsres.GetXaxis().SetTitleSize(0.04)
profditaumassnncorrabsres.GetXaxis().SetLabelSize(0.04)
profditaumassnncorrabsres.GetXaxis().SetTitleOffset(1.15)
profditaumassnncorrabsres.GetXaxis().SetLabelOffset(0.01)
profditaumassnncorrabsres.GetYaxis().SetTitleSize(0.04)
profditaumassnncorrabsres.GetYaxis().SetLabelSize(0.04)
profditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
profditaumassnncorrabsres.GetYaxis().SetLabelOffset(0.01)
profditaumassnncorrabsres.SetStats(0)
profditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres = ROOT.TH1D("histprofditaumassnncorrabsres","average of the absolute relative differences per event",300,50,350)
histprofditaumassnncorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histprofditaumassnncorrabsres.GetYaxis().SetTitle("average of the absolute relative differences per event")
histprofditaumassnncorrabsres.GetXaxis().SetTitleSize(0.04)
histprofditaumassnncorrabsres.GetXaxis().SetLabelSize(0.04)
histprofditaumassnncorrabsres.GetXaxis().SetTitleOffset(1.15)
histprofditaumassnncorrabsres.GetXaxis().SetLabelOffset(0.01)
histprofditaumassnncorrabsres.GetYaxis().SetTitleSize(0.04)
histprofditaumassnncorrabsres.GetYaxis().SetLabelSize(0.04)
histprofditaumassnncorrabsres.GetYaxis().SetTitleOffset(1.2)
histprofditaumassnncorrabsres.GetYaxis().SetLabelOffset(0.01)
histprofditaumassnncorrabsres.SetStats(0)
histprofditaumassnncorrabsres.SetLineColor(4)
histprofditaumassnncorrabsres.SetMarkerStyle(7)
histprofditaumassnncorrabsres.SetMarkerColor(4)
profditaumasssvfitcorrabsres = ROOT.TProfile("profditaumasssvfitcorrabsres","average of the absolute relative differences per event",300,50,350)
profditaumasssvfitcorrabsres.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
profditaumasssvfitcorrabsres.GetYaxis().SetTitle("average of the absolute relative difference per event")
profditaumasssvfitcorrabsres.GetXaxis().SetTitleSize(0.04)
profditaumasssvfitcorrabsres.GetXaxis().SetLabelSize(0.04)
profditaumasssvfitcorrabsres.GetXaxis().SetTitleOffset(1.15)
profditaumasssvfitcorrabsres.GetXaxis().SetLabelOffset(0.01)
profditaumasssvfitcorrabsres.GetYaxis().SetTitleSize(0.04)
profditaumasssvfitcorrabsres.GetYaxis().SetLabelSize(0.04)
profditaumasssvfitcorrabsres.GetYaxis().SetTitleOffset(1.2)
profditaumasssvfitcorrabsres.GetYaxis().SetLabelOffset(0.01)
profditaumasssvfitcorrabsres.SetStats(0)
profditaumasssvfitcorrabsres.SetMarkerStyle(7)
histprofditaumasssvfitcorrabsres = ROOT.TH1D("histprofditaumasssvfitcorrabsres","mean of |resolution| per di-#tau_{gen} mass",300,50,350)
histprofditaumasssvfitcorrabsres.SetStats(0)
histprofditaumasssvfitcorrabsres.SetLineColor(8)
histprofditaumasssvfitcorrabsres.SetMarkerStyle(7)
histprofditaumasssvfitcorrabsres.SetMarkerColor(8)

#ratio histograms
histditaumassnnratio = ROOT.TH1D("ditaumassregallratio","ratio between reconstruced and actual mass",60,50,350)
histditaumassnnratio.SetTitle("")
histditaumassnnratio.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnnratio.GetXaxis().SetLabelSize(0.12)
histditaumassnnratio.GetXaxis().SetTitleSize(0.12)
histditaumassnnratio.GetXaxis().SetLabelOffset(0.06)
histditaumassnnratio.GetXaxis().SetTitleOffset(1.3)
histditaumassnnratio.GetYaxis().SetTitle("ratio")
histditaumassnnratio.GetYaxis().SetLabelSize(0.13)
histditaumassnnratio.GetYaxis().SetTitleSize(0.13)
histditaumassnnratio.GetYaxis().SetTitleOffset(0.37)
histditaumassnnratio.GetYaxis().SetNdivisions(404)
histditaumassnnratio.GetYaxis().CenterTitle()
histditaumassnnratio.GetYaxis().SetRangeUser(0.0,2.0)
histditaumassnnratio.GetYaxis().SetLabelOffset(0.01)
histditaumassnnratio.SetMarkerStyle(7)
histditaumassnnratio.SetMarkerColor(4)
histditaumassnnratio.SetStats(0)
histditaumasssvfitratio = ROOT.TH1D("ditaumasssvfitratio","ratio between svfit and actual mass",60,50,350)
histditaumasssvfitratio.SetMarkerStyle(7)
histditaumasssvfitratio.SetMarkerColor(8)
histditaumasssvfitratio.SetStats(0)

#histogram of ditau mass using neural network and SVfit 100 GeV
histditaumass100GeV = ROOT.TH1D("ditaumass100GeV","reconstruct di-#tau mass using neural network and SVfit",100,70,130)
histditaumass100GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass100GeV.GetYaxis().SetTitle("number of occurence")
histditaumass100GeV.GetXaxis().SetTitleSize(0.04)
histditaumass100GeV.GetXaxis().SetLabelSize(0.04)
histditaumass100GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass100GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass100GeV.GetYaxis().SetTitleSize(0.04)
histditaumass100GeV.GetYaxis().SetLabelSize(0.04)
histditaumass100GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass100GeV.GetYaxis().SetTitleOffset(1.2)
histditaumass100GeV.SetLineColor(2)
histditaumass100GeV.SetLineWidth(3)
histditaumass100GeV.SetStats(0)
histditaumassnn100GeV = ROOT.TH1D("ditaumassnn100GeV","reconstruct di-#tau mass using neural network",100,70,130)
histditaumassnn100GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn100GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn100GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn100GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn100GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn100GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn100GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn100GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn100GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn100GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn100GeV.SetLineColor(4)
histditaumassnn100GeV.SetLineWidth(3)
histditaumassnn100GeV.SetLineStyle(7)
histditaumassnn100GeV.SetStats(0)
histditaumasssvfit100GeV = ROOT.TH1D("ditaumasssvfit100GeV","di-#tau mass using SVfit",100,70,130)
histditaumasssvfit100GeV.SetLineColor(3)
histditaumasssvfit100GeV.SetLineWidth(3)
histditaumasssvfit100GeV.SetLineStyle(2)
histditaumasssvfit100GeV.SetStats(0)
histditaumassnnres100GeV = ROOT.TH1D("NN100GeV","relative difference per event",80,-1,1)
histditaumassnnres100GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres100GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres100GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres100GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres100GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres100GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres100GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres100GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres100GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres100GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres100GeV.SetLineColor(4)
histditaumassnnres100GeV.SetLineWidth(3)
histditaumasssvfitres100GeV = ROOT.TH1D("SVfit100GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres100GeV.SetLineColor(8)
histditaumasssvfitres100GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 110 GeV
histditaumass110GeV = ROOT.TH1D("ditaumass110GeV","reconstruct di-#tau mass using neural network and SVfit",100,80,140)
histditaumass110GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass110GeV.GetYaxis().SetTitle("number of occurence")
histditaumass110GeV.GetXaxis().SetTitleSize(0.04)
histditaumass110GeV.GetXaxis().SetLabelSize(0.04)
histditaumass110GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass110GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass110GeV.GetYaxis().SetTitleSize(0.04)
histditaumass110GeV.GetYaxis().SetLabelSize(0.04)
histditaumass110GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass110GeV.SetLineColor(2)
histditaumass110GeV.SetLineWidth(3)
histditaumass110GeV.SetStats(0)
histditaumassnn110GeV = ROOT.TH1D("ditaumassnn110GeV","reconstruct di-#tau mass using neural network",100,80,140)
histditaumassnn110GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn110GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn110GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn110GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn110GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn110GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn110GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn110GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn110GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn110GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn110GeV.SetLineColor(4)
histditaumassnn110GeV.SetLineWidth(3)
histditaumassnn110GeV.SetLineStyle(7)
histditaumassnn110GeV.SetStats(0)
histditaumasssvfit110GeV = ROOT.TH1D("ditaumasssvfit110GeV","di-#tau mass using SVfit",100,80,140)
histditaumasssvfit110GeV.SetLineColor(3)
histditaumasssvfit110GeV.SetLineWidth(3)
histditaumasssvfit110GeV.SetLineStyle(2)
histditaumasssvfit110GeV.SetStats(0)
histditaumassnnres110GeV = ROOT.TH1D("NN110GeV","relative difference per event",80,-1,1)
histditaumassnnres110GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres110GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres110GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres110GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres110GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres110GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres110GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres110GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres110GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres110GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres110GeV.SetLineColor(4)
histditaumassnnres110GeV.SetLineWidth(3)
histditaumasssvfitres110GeV = ROOT.TH1D("SVfit110GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres110GeV.SetLineColor(8)
histditaumasssvfitres110GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 125 GeV
histditaumass125GeV = ROOT.TH1D("ditaumass125GeV","reconstruct di-#tau mass using neural network and SVfit",100,95,155)
histditaumass125GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass125GeV.GetYaxis().SetTitle("number of occurence")
histditaumass125GeV.GetXaxis().SetTitleSize(0.04)
histditaumass125GeV.GetXaxis().SetLabelSize(0.04)
histditaumass125GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass125GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass125GeV.GetYaxis().SetTitleSize(0.04)
histditaumass125GeV.GetYaxis().SetLabelSize(0.04)
histditaumass125GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass125GeV.GetYaxis().SetTitleOffset(1.2)
histditaumass125GeV.SetLineColor(2)
histditaumass125GeV.SetLineWidth(3)
histditaumass125GeV.SetStats(0)
histditaumassnn125GeV = ROOT.TH1D("ditaumassnn125GeV","reconstruct di-#tau mass using neural network",100,95,155)
histditaumassnn125GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn125GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn125GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn125GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn125GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn125GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn125GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn125GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn125GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn125GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn125GeV.SetLineColor(4)
histditaumassnn125GeV.SetLineWidth(3)
histditaumassnn125GeV.SetLineStyle(7)
histditaumassnn125GeV.SetStats(0)
histditaumasssvfit125GeV = ROOT.TH1D("ditaumasssvfit125GeV","di-#tau mass using SVfit",100,95,155)
histditaumasssvfit125GeV.SetLineColor(3)
histditaumasssvfit125GeV.SetLineWidth(3)
histditaumasssvfit125GeV.SetLineStyle(2)
histditaumasssvfit125GeV.SetStats(0)
histditaumassnnres125GeV = ROOT.TH1D("NN125GeV","relative difference per event",80,-1,1)
histditaumassnnres125GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres125GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres125GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres125GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres125GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres125GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres125GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres125GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres125GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres125GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres125GeV.SetLineColor(4)
histditaumassnnres125GeV.SetLineWidth(3)
histditaumasssvfitres125GeV = ROOT.TH1D("SVfit125GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres125GeV.SetLineColor(8)
histditaumasssvfitres125GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 140 GeV
histditaumass140GeV = ROOT.TH1D("ditaumass140GeV","reconstruct di-#tau mass using neural network and SVfit",100,110,170)
histditaumass140GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass140GeV.GetYaxis().SetTitle("number of occurence")
histditaumass140GeV.GetXaxis().SetTitleSize(0.04)
histditaumass140GeV.GetXaxis().SetLabelSize(0.04)
histditaumass140GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass140GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass140GeV.GetYaxis().SetTitleSize(0.04)
histditaumass140GeV.GetYaxis().SetLabelSize(0.04)
histditaumass140GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass140GeV.GetYaxis().SetTitleOffset(1.2)
histditaumass140GeV.SetLineColor(2)
histditaumass140GeV.SetLineWidth(3)
histditaumass140GeV.SetStats(0)
histditaumassnn140GeV = ROOT.TH1D("ditaumassnn140GeV","reconstruct di-#tau mass using neural network",100,110,170)
histditaumassnn140GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn140GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn140GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn140GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn140GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn140GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn140GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn140GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn140GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn140GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn140GeV.SetLineColor(4)
histditaumassnn140GeV.SetLineWidth(3)
histditaumassnn140GeV.SetLineStyle(7)
histditaumassnn140GeV.SetStats(0)
histditaumasssvfit140GeV = ROOT.TH1D("ditaumasssvfit140GeV","di-#tau mass using SVfit",100,110,170)
histditaumasssvfit140GeV.SetLineColor(3)
histditaumasssvfit140GeV.SetLineWidth(3)
histditaumasssvfit140GeV.SetLineStyle(2)
histditaumasssvfit140GeV.SetStats(0)
histditaumassnnres140GeV = ROOT.TH1D("NN140GeV","relative difference per event",80,-1,1)
histditaumassnnres140GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres140GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres140GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres140GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres140GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres140GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres140GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres140GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres140GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres140GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres140GeV.SetLineColor(4)
histditaumassnnres140GeV.SetLineWidth(3)
histditaumasssvfitres140GeV = ROOT.TH1D("SVfit140GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres140GeV.SetLineColor(8)
histditaumasssvfitres140GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit Drell-Yan
histditaumassdy = ROOT.TH1D("ditaumassdy","reconstruct di-#tau mass using neural network and SVfit",100,60,120)
histditaumassdy.SetLineColor(2)
histditaumassdy.SetLineWidth(3)
histditaumassdy.SetStats(0)
histditaumassnndy = ROOT.TH1D("ditaumassnndy","reconstruct di-#tau mass using neural network",100,60,120)
histditaumassnndy.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnndy.GetYaxis().SetTitle("number of occurence")
histditaumassnndy.GetXaxis().SetTitleSize(0.04)
histditaumassnndy.GetXaxis().SetLabelSize(0.04)
histditaumassnndy.GetXaxis().SetTitleOffset(1.15)
histditaumassnndy.GetXaxis().SetLabelOffset(0.01)
histditaumassnndy.GetYaxis().SetTitleSize(0.04)
histditaumassnndy.GetYaxis().SetLabelSize(0.04)
histditaumassnndy.GetYaxis().SetLabelOffset(0.01)
histditaumassnndy.GetYaxis().SetTitleOffset(1.2)
histditaumassnndy.SetLineColor(4)
histditaumassnndy.SetLineWidth(3)
histditaumassnndy.SetLineStyle(7)
histditaumassnndy.SetStats(0)
histditaumasssvfitdy = ROOT.TH1D("ditaumasssvfitdy","di-#tau mass using SVfit",100,60,120)
histditaumasssvfitdy.SetLineColor(8)
histditaumasssvfitdy.SetLineWidth(3)
histditaumasssvfitdy.SetLineStyle(2)
histditaumasssvfitdy.SetStats(0)
histditaumassnnresdy = ROOT.TH1D("NNdy","relative difference per event",80,-1,1)
histditaumassnnresdy.GetXaxis().SetTitle("relative difference per event")
histditaumassnnresdy.GetYaxis().SetTitle("number of occurence")
histditaumassnnresdy.GetXaxis().SetTitleSize(0.04)
histditaumassnnresdy.GetXaxis().SetLabelSize(0.04)
histditaumassnnresdy.GetXaxis().SetTitleOffset(1.15)
histditaumassnnresdy.GetXaxis().SetLabelOffset(0.01)
histditaumassnnresdy.GetYaxis().SetTitleSize(0.04)
histditaumassnnresdy.GetYaxis().SetLabelSize(0.04)
histditaumassnnresdy.GetYaxis().SetLabelOffset(0.01)
histditaumassnnresdy.GetYaxis().SetTitleOffset(1.2)
histditaumassnnresdy.SetLineColor(4)
histditaumassnnresdy.SetLineWidth(3)
histditaumasssvfitresdy = ROOT.TH1D("SVfitdy","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitresdy.SetLineColor(8)
histditaumasssvfitresdy.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 110, 125 and 140 GeV and DY in same plot
histditaumass100GeVcomp = ROOT.TH1D("ditaumass100GeVcomp","reconstruct di-#tau mass using neural network and SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumass100GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass100GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumass100GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumass100GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumass100GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumass100GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumass100GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumass100GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumass100GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumass100GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumass100GeVcomp.SetLineColor(28)
histditaumass100GeVcomp.SetLineWidth(3)
histditaumass100GeVcomp.SetStats(0)
histditaumassnn100GeVcomp = ROOT.TH1D("ditaumassnn100GeVcomp","reconstruct di-#tau mass using neural network",100,signal_bkg_left,signal_bkg_right)
histditaumassnn100GeVcomp.SetLineColor(3)
histditaumassnn100GeVcomp.SetLineWidth(3)
histditaumassnn100GeVcomp.SetLineStyle(7)
histditaumassnn100GeVcomp.SetStats(0)
histditaumasssvfit100GeVcomp = ROOT.TH1D("ditaumasssvfit100GeVcomp","di-#tau mass using SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumasssvfit100GeVcomp.SetLineColor(6)
histditaumasssvfit100GeVcomp.SetLineWidth(3)
histditaumasssvfit100GeVcomp.SetLineStyle(2)
histditaumasssvfit100GeVcomp.SetStats(0)
histditaumass110GeVcomp = ROOT.TH1D("ditaumass110GeVcomp","reconstruct di-#tau mass using neural network and SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumass110GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass110GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumass110GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumass110GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumass110GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumass110GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumass110GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumass110GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumass110GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumass110GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumass110GeVcomp.SetLineColor(28)
histditaumass110GeVcomp.SetLineWidth(3)
histditaumass110GeVcomp.SetStats(0)
histditaumassnn110GeVcomp = ROOT.TH1D("ditaumassnn110GeVcomp","reconstruct di-#tau mass using neural network",100,signal_bkg_left,signal_bkg_right)
histditaumassnn110GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn110GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumassnn110GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumassnn110GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumassnn110GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumassnn110GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumassnn110GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumassnn110GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumassnn110GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumassnn110GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumassnn110GeVcomp.SetLineColor(3)
histditaumassnn110GeVcomp.SetLineWidth(3)
histditaumassnn110GeVcomp.SetLineStyle(7)
histditaumassnn110GeVcomp.SetStats(0)
histditaumasssvfit110GeVcomp = ROOT.TH1D("ditaumasssvfit110GeVcomp","di-#tau mass using SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumasssvfit110GeVcomp.SetLineColor(6)
histditaumasssvfit110GeVcomp.SetLineWidth(3)
histditaumasssvfit110GeVcomp.SetLineStyle(2)
histditaumasssvfit110GeVcomp.SetStats(0)
histditaumass125GeVcomp = ROOT.TH1D("ditaumass125GeVcomp","reconstruct di-#tau mass using neural network and SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumass125GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass125GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumass125GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumass125GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumass125GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumass125GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumass125GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumass125GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumass125GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumass125GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumass125GeVcomp.SetLineColor(28)
histditaumass125GeVcomp.SetLineWidth(3)
histditaumass125GeVcomp.SetStats(0)
histditaumassnn125GeVcomp = ROOT.TH1D("ditaumassnn125GeVcomp","reconstruct di-#tau mass using neural network",100,signal_bkg_left,signal_bkg_right)
histditaumassnn125GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn125GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumassnn125GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumassnn125GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumassnn125GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumassnn125GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumassnn125GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumassnn125GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumassnn125GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumassnn125GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumassnn125GeVcomp.SetLineColor(3)
histditaumassnn125GeVcomp.SetLineWidth(3)
histditaumassnn125GeVcomp.SetLineStyle(7)
histditaumassnn125GeVcomp.SetStats(0)
histditaumasssvfit125GeVcomp = ROOT.TH1D("ditaumasssvfit125GeVcomp","di-#tau mass using SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumasssvfit125GeVcomp.SetLineColor(6)
histditaumasssvfit125GeVcomp.SetLineWidth(3)
histditaumasssvfit125GeVcomp.SetLineStyle(2)
histditaumasssvfit125GeVcomp.SetStats(0)
#####################################
histditaumass140GeVcomp = ROOT.TH1D("ditaumass140GeVcomp","reconstruct di-#tau mass using neural network and SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumass140GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass140GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumass140GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumass140GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumass140GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumass140GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumass140GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumass140GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumass140GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumass140GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumass140GeVcomp.SetLineColor(28)
histditaumass140GeVcomp.SetLineWidth(3)
histditaumass140GeVcomp.SetStats(0)
histditaumassnn140GeVcomp = ROOT.TH1D("ditaumassnn140GeVcomp","reconstruct di-#tau mass using neural network",100,signal_bkg_left,signal_bkg_right)
histditaumassnn140GeVcomp.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn140GeVcomp.GetYaxis().SetTitle("number of occurence")
histditaumassnn140GeVcomp.GetXaxis().SetTitleSize(0.04)
histditaumassnn140GeVcomp.GetXaxis().SetLabelSize(0.04)
histditaumassnn140GeVcomp.GetXaxis().SetTitleOffset(1.15)
histditaumassnn140GeVcomp.GetXaxis().SetLabelOffset(0.01)
histditaumassnn140GeVcomp.GetYaxis().SetTitleSize(0.04)
histditaumassnn140GeVcomp.GetYaxis().SetLabelSize(0.04)
histditaumassnn140GeVcomp.GetYaxis().SetLabelOffset(0.01)
histditaumassnn140GeVcomp.GetYaxis().SetTitleOffset(1.2)
histditaumassnn140GeVcomp.SetLineColor(3)
histditaumassnn140GeVcomp.SetLineWidth(3)
histditaumassnn140GeVcomp.SetLineStyle(7)
histditaumassnn140GeVcomp.SetStats(0)
histditaumasssvfit140GeVcomp = ROOT.TH1D("ditaumasssvfit140GeVcomp","di-#tau mass using SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumasssvfit140GeVcomp.SetLineColor(6)
histditaumasssvfit140GeVcomp.SetLineWidth(3)
histditaumasssvfit140GeVcomp.SetLineStyle(2)
histditaumasssvfit140GeVcomp.SetStats(0)
histditaumassdycomp = ROOT.TH1D("ditaumassdycomp","reconstruct di-#tau mass using neural network and SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumassdycomp.SetLineColor(2)
histditaumassdycomp.SetLineWidth(3)
histditaumassdycomp.SetStats(0)
histditaumassnndycomp = ROOT.TH1D("ditaumassnndycomp","reconstruct di-#tau mass using neural network",100,signal_bkg_left,signal_bkg_right)
histditaumassnndycomp.SetLineColor(4)
histditaumassnndycomp.SetLineWidth(3)
histditaumassnndycomp.SetLineStyle(7)
histditaumassnndycomp.SetStats(0)
histditaumasssvfitdycomp = ROOT.TH1D("ditaumasssvfitdycomp","di-#tau mass using SVfit",100,signal_bkg_left,signal_bkg_right)
histditaumasssvfitdycomp.SetLineColor(1)
histditaumasssvfitdycomp.SetLineWidth(3)
histditaumasssvfitdycomp.SetLineStyle(2)
histditaumasssvfitdycomp.SetStats(0)

#histogram of ditau mass using neural network and SVfit 250 GeV
histditaumass250GeV = ROOT.TH1D("ditaumass250GeV","reconstruct di-#tau mass using neural network and SVfit",100,220,280)
histditaumass250GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass250GeV.GetYaxis().SetTitle("number of occurence")
histditaumass250GeV.GetXaxis().SetTitleSize(0.04)
histditaumass250GeV.GetXaxis().SetLabelSize(0.04)
histditaumass250GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass250GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass250GeV.GetYaxis().SetTitleSize(0.04)
histditaumass250GeV.GetYaxis().SetLabelSize(0.04)
histditaumass250GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass250GeV.GetYaxis().SetTitleOffset(1.2)
histditaumass250GeV.SetLineColor(2)
histditaumass250GeV.SetLineWidth(3)
histditaumass250GeV.SetStats(0)
histditaumassnn250GeV = ROOT.TH1D("ditaumassnn250GeV","reconstructed di-#tau mass using neural network",100,220,280)
histditaumassnn250GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn250GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn250GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn250GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn250GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn250GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn250GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn250GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn250GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn250GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn250GeV.SetLineColor(4)
histditaumassnn250GeV.SetLineWidth(3)
histditaumassnn250GeV.SetLineStyle(7)
histditaumassnn250GeV.SetStats(0)
histditaumasssvfit250GeV = ROOT.TH1D("ditaumasssvfit250GeV","di-#tau mass using SVfit",100,220,280)
histditaumasssvfit250GeV.SetLineColor(8)
histditaumasssvfit250GeV.SetLineWidth(3)
histditaumasssvfit250GeV.SetLineStyle(2)
histditaumasssvfit250GeV.SetStats(0)
histditaumassnnres250GeV = ROOT.TH1D("NN250GeV","relative difference per event",80,-1,1)
histditaumassnnres250GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres250GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres250GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres250GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres250GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres250GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres250GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres250GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres250GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres250GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres250GeV.SetLineColor(4)
histditaumassnnres250GeV.SetLineWidth(3)
histditaumasssvfitres250GeV = ROOT.TH1D("SVfit250GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres250GeV.SetLineColor(8)
histditaumasssvfitres250GeV.SetLineWidth(3)

#histogram of ditau mass using neural network and SVfit 180 GeV
histditaumass180GeV = ROOT.TH1D("ditaumass180GeV","reconstruct di-#tau mass using neural network and SVfit",100,150,210)
histditaumass180GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumass180GeV.GetYaxis().SetTitle("number of occurence")
histditaumass180GeV.GetXaxis().SetTitleSize(0.04)
histditaumass180GeV.GetXaxis().SetLabelSize(0.04)
histditaumass180GeV.GetXaxis().SetTitleOffset(1.15)
histditaumass180GeV.GetXaxis().SetLabelOffset(0.01)
histditaumass180GeV.GetYaxis().SetTitleSize(0.04)
histditaumass180GeV.GetYaxis().SetLabelSize(0.04)
histditaumass180GeV.GetYaxis().SetLabelOffset(0.01)
histditaumass180GeV.GetYaxis().SetTitleOffset(1.2)
histditaumass180GeV.SetLineColor(2)
histditaumass180GeV.SetLineWidth(3)
histditaumass180GeV.SetStats(0)
histditaumassnn180GeV = ROOT.TH1D("ditaumassnn180GeV","reconstructed di-#tau mass using neural network",100,150,210)
histditaumassnn180GeV.GetXaxis().SetTitle("di-#tau_{gen} mass [GeV]")
histditaumassnn180GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnn180GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnn180GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnn180GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnn180GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnn180GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnn180GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnn180GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnn180GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnn180GeV.SetLineColor(4)
histditaumassnn180GeV.SetLineWidth(3)
histditaumassnn180GeV.SetLineStyle(7)
histditaumassnn180GeV.SetStats(0)
histditaumasssvfit180GeV = ROOT.TH1D("ditaumasssvfit180GeV","di-#tau mass using SVfit",100,150,210)
histditaumasssvfit180GeV.SetLineColor(8)
histditaumasssvfit180GeV.SetLineWidth(3)
histditaumasssvfit180GeV.SetLineStyle(2)
histditaumasssvfit180GeV.SetStats(0)
histditaumassnnres180GeV = ROOT.TH1D("NN180GeV","relative difference per event",80,-1,1)
histditaumassnnres180GeV.GetXaxis().SetTitle("relative difference per event")
histditaumassnnres180GeV.GetYaxis().SetTitle("number of occurence")
histditaumassnnres180GeV.GetXaxis().SetTitleSize(0.04)
histditaumassnnres180GeV.GetXaxis().SetLabelSize(0.04)
histditaumassnnres180GeV.GetXaxis().SetTitleOffset(1.15)
histditaumassnnres180GeV.GetXaxis().SetLabelOffset(0.01)
histditaumassnnres180GeV.GetYaxis().SetTitleSize(0.04)
histditaumassnnres180GeV.GetYaxis().SetLabelSize(0.04)
histditaumassnnres180GeV.GetYaxis().SetLabelOffset(0.01)
histditaumassnnres180GeV.GetYaxis().SetTitleOffset(1.2)
histditaumassnnres180GeV.SetLineColor(4)
histditaumassnnres180GeV.SetLineWidth(3)
histditaumasssvfitres180GeV = ROOT.TH1D("SVfit180GeV","relative difference per event using SVfit",80,-1,1)
histditaumasssvfitres180GeV.SetLineColor(8)
histditaumasssvfitres180GeV.SetLineWidth(3)

##############      preparing the histograms        #################
def prep_histograms(epochs):
    for j in ditaumass_nn:
        histditaumassnn.Fill(j)
    for s,ditaumass100GeV_value in enumerate(test_ditaumass_100GeV):
        histditaumassnn100GeV.Fill(ditaumass_nn_100GeV[s])
        histditaumassnn100GeVcomp.Fill(ditaumass_nn_100GeV[s])
        res = (ditaumass100GeV_value - ditaumass_nn_100GeV[s])/ditaumass100GeV_value
        histditaumassnnres100GeV.Fill(res)
    for t,ditaumass110GeV_value in enumerate(test_ditaumass_110GeV):
        histditaumassnn110GeV.Fill(ditaumass_nn_110GeV[t])
        histditaumassnn110GeVcomp.Fill(ditaumass_nn_110GeV[t])
        res = (ditaumass110GeV_value - ditaumass_nn_110GeV[t])/ditaumass110GeV_value
        histditaumassnnres110GeV.Fill(res)
    for s,ditaumass125GeV_value in enumerate(test_ditaumass_125GeV):
        histditaumassnn125GeV.Fill(ditaumass_nn_125GeV[s])
        histditaumassnn125GeVcomp.Fill(ditaumass_nn_125GeV[s])
        res = (ditaumass125GeV_value - ditaumass_nn_125GeV[s])/ditaumass125GeV_value
        histditaumassnnres125GeV.Fill(res)
    for s,ditaumass140GeV_value in enumerate(test_ditaumass_140GeV):
        histditaumassnn140GeV.Fill(ditaumass_nn_140GeV[s])
        histditaumassnn140GeVcomp.Fill(ditaumass_nn_140GeV[s])
        res = (ditaumass140GeV_value - ditaumass_nn_140GeV[s])/ditaumass140GeV_value
        histditaumassnnres140GeV.Fill(res)
    for y,ditaumass180GeV_value in enumerate(test_ditaumass_180GeV):
        histditaumassnn180GeV.Fill(ditaumass_nn_180GeV[y])
        res = (ditaumass180GeV_value - ditaumass_nn_180GeV[y])/ditaumass180GeV_value
        histditaumassnnres180GeV.Fill(res)
    for x,ditaumass250GeV_value in enumerate(test_ditaumass_250GeV):
        histditaumassnn250GeV.Fill(ditaumass_nn_250GeV[x])
        res = (ditaumass250GeV_value - ditaumass_nn_250GeV[x])/ditaumass250GeV_value
        histditaumassnnres250GeV.Fill(res)
    for r,ditaumassdy_value in enumerate(test_ditaumass_dy):
        histditaumassnndy.Fill(ditaumass_nn_dy[r])
        histditaumassnndycomp.Fill(ditaumass_nn_dy[r])
        res = (ditaumassdy_value - ditaumass_nn_dy[r])/ditaumassdy_value
        histditaumassnnresdy.Fill(res)
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        histditaumassnnres.Fill(res)
        histditaumassnnrescomp.Fill(res)
        histditaumassnncorrres.Fill(ditaumass_value,res)
        profditaumassnncorrres.Fill(ditaumass_value,res)
        #histditaumassnncorrres.Fill(ditaumass_nn[i],res)
        #profditaumassnncorrres.Fill(ditaumass_nn[i],res)
        profditaumassnncorrabsres.Fill(ditaumass_value,abs(res))
    for g in range(len(ditaumass_nn)):
        histditaumassnncorr.Fill(test_ditaumass_selected[g],ditaumass_nn[g])
        profditaumassnncorrrms.Fill(test_ditaumass_selected[g],ditaumass_nn[g])
    for j in range(300):
        rms = profditaumassnncorrrms.GetBinError(j+1)
        histditaumassnnrms.SetBinContent(j+1,rms)
    for k in range(60):
        if histditaumass.GetBinContent(k+1) != 0:
            content_nn = histditaumassnn.GetBinContent(k+1)
            content_actual = histditaumass.GetBinContent(k+1)
            ratio = content_nn/content_actual
            error_nn = numpy.sqrt(content_nn)
            error_actual = numpy.sqrt(content_actual)
            error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_nn/content_nn)**2)
            histditaumassnnratio.SetBinError(k+1,error_ratio)
            histditaumassnnratio.SetBinContent(k+1,ratio)
        elif histditaumassnn.GetBinContent(k+1) == 0 and histditaumass.GetBinContent(k+1) == 0:
            histditaumassnnratio.SetBinContent(k+1,1.0)
    epochs_range = numpy.array([float(i) for i in range(1,epochs+1)])
    loss_graph = ROOT.TGraph(epochs,epochs_range,loss_values)
    loss_graph.SetTitle("model loss")
    loss_graph.GetXaxis().SetTitle("epochs")
    loss_graph.GetYaxis().SetTitle("loss")
    loss_graph.SetMarkerColor(4)
    loss_graph.SetMarkerSize(0.8)
    loss_graph.SetMarkerStyle(21)
    val_loss_graph = ROOT.TGraph(epochs,epochs_range,val_loss_values)
    val_loss_graph.SetMarkerColor(2)
    val_loss_graph.SetMarkerSize(0.8)
    val_loss_graph.SetMarkerStyle(21)
    canv1 = ROOT.TCanvas("loss di-tau mass")
    loss_graph.Draw("AP")
    val_loss_graph.Draw("P")
    leg1 = ROOT.TLegend(0.6,0.7,0.87,0.87)
    leg1.AddEntry(loss_graph,"loss on train sample","P")
    leg1.AddEntry(val_loss_graph,"loss on test sample","P")
    leg1.Draw()
    output_plot_name = "%s_loss.png" %(output_name)
    canv1.SaveAs(output_plot_name)


def prep_svfit_histograms():
    for i,ditaumass_value in enumerate(ditaumass_svfit_gen):
        histditaumasssvfit.Fill(ditaumass_svfit[i])
        res = (ditaumass_value - ditaumass_svfit[i])/ditaumass_value
        histditaumasssvfitres.Fill(res)
        histditaumasssvfitcorrres.Fill(ditaumass_value,res)
        profditaumasssvfitcorrres.Fill(ditaumass_value,res)
        #histditaumasssvfitcorrres.Fill(ditaumass_svfit[i],res)
        #profditaumasssvfitcorrres.Fill(ditaumass_svfit[i],res)
        profditaumasssvfitcorrabsres.Fill(ditaumass_value,abs(res))
    for g in range(len(ditaumass_svfit)):
        histditaumasssvfitcorr.Fill(ditaumass_svfit_gen[g],ditaumass_svfit[g])
    for k in range(60):
        if histditaumass.GetBinContent(k+1) != 0:
            content_svfit = histditaumasssvfit.GetBinContent(k+1)
            content_actual = histditaumass.GetBinContent(k+1)
            ratio = content_svfit/content_actual
            error_svfit = numpy.sqrt(content_svfit)
            error_actual = numpy.sqrt(content_actual)
            error_ratio = ratio*numpy.sqrt((error_actual/content_actual)**2+(error_svfit/content_svfit)**2)
            histditaumasssvfitratio.SetBinError(k+1,error_ratio)
            histditaumasssvfitratio.SetBinContent(k+1,ratio)
        elif histditaumasssvfit.GetBinContent(k+1) == 0 and histditaumass.GetBinContent(k+1) == 0:
            histditaumasssvfitratio.SetBinContent(k+1,1.0)
    for u,ditaumass100GeV_value in enumerate(ditaumass_svfit100GeV_gen):
        histditaumasssvfit100GeV.Fill(ditaumass_svfit100GeV[u])
        histditaumasssvfit100GeVcomp.Fill(ditaumass_svfit100GeV[u])
        res = (ditaumass100GeV_value - ditaumass_svfit100GeV[u])/ditaumass100GeV_value
        histditaumasssvfitres100GeV.Fill(res)
    for d,ditaumass110GeV_value in enumerate(ditaumass_svfit110GeV_gen):
        histditaumasssvfit110GeV.Fill(ditaumass_svfit110GeV[d])
        histditaumasssvfit110GeVcomp.Fill(ditaumass_svfit110GeV[d])
        res = (ditaumass110GeV_value - ditaumass_svfit110GeV[d])/ditaumass110GeV_value
        histditaumasssvfitres110GeV.Fill(res)
    for j,ditaumass125GeV_value in enumerate(ditaumass_svfit125GeV_gen):
        histditaumasssvfit125GeV.Fill(ditaumass_svfit125GeV[j])
        histditaumasssvfit125GeVcomp.Fill(ditaumass_svfit125GeV[j])
        res = (ditaumass125GeV_value - ditaumass_svfit125GeV[j])/ditaumass125GeV_value
        histditaumasssvfitres125GeV.Fill(res)
    for g,ditaumass140GeV_value in enumerate(ditaumass_svfit140GeV_gen):
        histditaumasssvfit140GeV.Fill(ditaumass_svfit140GeV[g])
        histditaumasssvfit140GeVcomp.Fill(ditaumass_svfit140GeV[g])
        res = (ditaumass140GeV_value - ditaumass_svfit140GeV[g])/ditaumass140GeV_value
        histditaumasssvfitres140GeV.Fill(res)
    for f,ditaumassdy_value in enumerate(ditaumass_svfitdy_gen):
        histditaumasssvfitdy.Fill(ditaumass_svfitdy[f])
        histditaumasssvfitdycomp.Fill(ditaumass_svfitdy[f])
        res = (ditaumassdy_value - ditaumass_svfitdy[f])/ditaumassdy_value
        histditaumasssvfitresdy.Fill(res)
    for w,ditaumass180GeV_value in enumerate(ditaumass_svfit180GeV_gen):
        histditaumasssvfit180GeV.Fill(ditaumass_svfit180GeV[w])
        res = (ditaumass180GeV_value - ditaumass_svfit180GeV[w])/ditaumass180GeV_value
        histditaumasssvfitres180GeV.Fill(res)
    for s,ditaumass250GeV_value in enumerate(ditaumass_svfit250GeV_gen):
        histditaumasssvfit250GeV.Fill(ditaumass_svfit250GeV[s])
        res = (ditaumass250GeV_value - ditaumass_svfit250GeV[s])/ditaumass250GeV_value
        histditaumasssvfitres250GeV.Fill(res)



def correct_genbias():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        profditaumassnncorrresbefore.Fill(ditaumass_value,res)
    fitfunc_nn = ROOT.TF1("fitfunc_nn","pol 3",50,350)
    profditaumassnncorrresbefore.Fit(fitfunc_nn)
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        nn_bias = fitfunc_nn.Eval(ditaumass_value)
        nn_oldvalue = ditaumass_nn[i]
        nn_newvalue = nn_oldvalue*(1.0+nn_bias)
        ditaumass_nn[i] = nn_newvalue
    for f,ditaumass_value in enumerate(ditaumass_svfit_gen):
        ditaumass_svfit_value = ditaumass_svfit[f]
        res = (ditaumass_value - ditaumass_svfit_value)/ditaumass_value
        profditaumasssvfitcorrresbefore.Fill(ditaumass_value,res)
    fitfunc_svfit = ROOT.TF1("fitfunc_svfit","pol 2",50,350)
    profditaumasssvfitcorrresbefore.Fit(fitfunc_svfit)
    for j, ditaumass_value in enumerate(ditaumass_svfit_gen):
        svfit_bias = fitfunc_svfit.Eval(ditaumass_value)
        svfit_oldvalue = ditaumass_svfit[j]
        svfit_newvalue = svfit_oldvalue*(1.0+svfit_bias)
        ditaumass_svfit[j] = svfit_newvalue

def correct_bias():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        res = (ditaumass_value - ditaumass_nn[i])/ditaumass_value
        profditaumassnncorrresbefore.Fill(ditaumass_nn[i],res)
    fitfunc_nn = ROOT.TF1("fitfunc_nn","pol 5",50,450)
    profditaumassnncorrresbefore.Fit(fitfunc_nn)
    for i,ditaumass_value in enumerate(ditaumass_nn):
        nn_bias = fitfunc_nn.Eval(ditaumass_value)
        nn_oldvalue = ditaumass_nn[i]
        nn_newvalue = nn_oldvalue*(1.0+nn_bias)
        ditaumass_nn[i] = nn_newvalue
    for f,ditaumass_value in enumerate(ditaumass_svfit_gen):
        ditaumass_svfit_value = ditaumass_svfit[f]
        res = (ditaumass_value - ditaumass_svfit_value)/ditaumass_value
        profditaumasssvfitcorrresbefore.Fill(ditaumass_svfit_value,res)
    fitfunc_svfit = ROOT.TF1("fitfunc_svfit","pol 7",50,450)
    profditaumasssvfitcorrresbefore.Fit(fitfunc_svfit)
    for j, ditaumass_value in enumerate(ditaumass_svfit):
        svfit_bias = fitfunc_svfit.Eval(ditaumass_value)
        svfit_oldvalue = ditaumass_svfit[j]
        svfit_newvalue = svfit_oldvalue*(1.0+svfit_bias)
        ditaumass_svfit[j] = svfit_newvalue


def fill_histditaumass():
    for i,ditaumass_value in enumerate(test_ditaumass):
        histditaumass.Fill(ditaumass_value)

def fill_histditauvismass():
    for f,ditauvismass_value in enumerate(test_ditauvismass):
        histditauvismass.Fill(ditauvismass_value)

def fill_histditaucollinearmass():
    for d,ditaucollinearmass_value in enumerate(test_ditaucollinearmass):
        histditaucollmass.Fill(ditaucollinearmass_value)

def fill_histditaumass_selected():
    for i,ditaumass_value in enumerate(test_ditaumass_selected):
        histditaumass.Fill(ditaumass_value)
        histditaumassgen.Fill(ditaumass_value)

def fill_histditauvismass_selected():
    for i,ditauvismass_value in enumerate(test_ditauvismass_selected):
        histditauvismass.Fill(ditauvismass_value)

def fill_histditaucollinearmass_selected():
    for d,ditaucollinearmass_value in enumerate(test_ditaucollinearmass_selected):
        histditaucollmass.Fill(ditaucollinearmass_value)

def fill_histprofres():
    for h in range(350):
        corrres = profditaumassnncorrres.GetBinContent(h+1)
        corrres_error = profditaumassnncorrres.GetBinError(h+1)
        histprofditaumassnncorrres.SetBinContent(h+1,corrres)
        histprofditaumassnncorrres.SetBinError(h+1,corrres_error)

def fill_histprofabsres():
    for i in range(350):
        corrabsres = profditaumassnncorrabsres.GetBinContent(i+1)
        corrabsres_error = profditaumassnncorrabsres.GetBinError(i+1)
        histprofditaumassnncorrabsres.SetBinContent(i+1,corrabsres)
        histprofditaumassnncorrabsres.SetBinError(i+1,corrabsres_error)
def fill_svfit_histprofres():
    for h in range(350):
        corrres = profditaumasssvfitcorrres.GetBinContent(h+1)
        corrres_error = profditaumasssvfitcorrres.GetBinError(h+1)
        histprofditaumasssvfitcorrres.SetBinContent(h+1,corrres)
        histprofditaumasssvfitcorrres.SetBinError(h+1,corrres_error)

def fill_svfit_histprofabsres():
    for i in range(350):
        corrabsres = profditaumasssvfitcorrabsres.GetBinContent(i+1)
        corrabsres_error = profditaumasssvfitcorrabsres.GetBinError(i+1)
        histprofditaumasssvfitcorrabsres.SetBinContent(i+1,corrabsres)
        histprofditaumasssvfitcorrabsres.SetBinError(i+1,corrabsres_error)

def fill_histditaumass_100GeV():
    for i,ditaumass100GeV_value in enumerate(test_ditaumass_100GeV):
        histditaumass100GeV.Fill(ditaumass100GeV_value)
        histditaumass100GeVcomp.Fill(ditaumass100GeV_value)
def fill_histditaumass_110GeV():
    for i,ditaumass110GeV_value in enumerate(test_ditaumass_110GeV):
        histditaumass110GeV.Fill(ditaumass110GeV_value)
        histditaumass110GeVcomp.Fill(ditaumass110GeV_value)
def fill_histditaumass_125GeV():
    for i,ditaumass125GeV_value in enumerate(test_ditaumass_125GeV):
        histditaumass125GeV.Fill(ditaumass125GeV_value)
        histditaumass125GeVcomp.Fill(ditaumass125GeV_value)
def fill_histditaumass_140GeV():
    for i,ditaumass140GeV_value in enumerate(test_ditaumass_140GeV):
        histditaumass140GeV.Fill(ditaumass140GeV_value)
        histditaumass140GeVcomp.Fill(ditaumass140GeV_value)
def fill_histditaumass_180GeV():
    for i,ditaumass180GeV_value in enumerate(test_ditaumass_180GeV):
        histditaumass180GeV.Fill(ditaumass180GeV_value)
def fill_histditaumass_250GeV():
    for i,ditaumass250GeV_value in enumerate(test_ditaumass_250GeV):
        histditaumass250GeV.Fill(ditaumass250GeV_value)
def fill_histditaumass_dy():
    for i,ditaumassdy_value in enumerate(test_ditaumass_dy):
        histditaumassdy.Fill(ditaumassdy_value)
        histditaumassdycomp.Fill(ditaumassdy_value)


###################  fill histograms  ################################################
fill_histditaumass_selected()
fill_histditauvismass_selected()
fill_histditaucollinearmass_selected()
if bias_correction == "reco":
    correct_bias()
if bias_correction == "gen":
    correct_genbias()
fill_histditaumass_100GeV()
fill_histditaumass_110GeV()
fill_histditaumass_125GeV()
fill_histditaumass_140GeV()
fill_histditaumass_180GeV()
fill_histditaumass_250GeV()
fill_histditaumass_dy()
prep_histograms(epochs)
prep_svfit_histograms()
fill_histprofres()
fill_histprofabsres()
fill_svfit_histprofres()
fill_svfit_histprofabsres()

####################        calculate signal over background        #############################
def integral(histo,minmass,maxmass):
    minBin = histo.GetXaxis().FindBin(minmass)
    maxBin = histo.GetXaxis().FindBin(maxmass)
    nentriesErr = ROOT.Double()
    nentries = histo.IntegralAndError(minBin,maxBin,nentriesErr)
    return [nentries,nentriesErr]


def scale(hist):
    integral = hist.Integral()
    hist.Scale(1./integral)

def efficiency(hist,signal_left,signal_right,number_test_values):
    return integral(hist,signal_left,signal_right)[0]/number_test_values

def efficiency_comp(eff_sig_nn,eff_bkg_nn,eff_sig_svfit,eff_bkg_svfit):
    return (eff_sig_nn/numpy.sqrt(eff_bkg_nn))/(eff_sig_svfit/numpy.sqrt(eff_bkg_svfit))

def signal_to_background(eff_sig,cross_section_sig,eff_bkg,cross_section_bkg,luminosity):
    return numpy.sqrt(luminosity)*eff_sig*cross_section_sig/numpy.sqrt(eff_bkg*cross_section_bkg)

def efficiency_error(hist,signal_left,signal_right,number_test_values):
    return integral(hist,signal_left,signal_right)[1]/number_test_values 

def signal_to_background_error(eff_sig,eff_sig_error,cross_section_sig,cross_section_sig_error,eff_bkg,eff_bkg_error,cross_section_bkg,cross_section_bkg_error,luminosity):
    return numpy.sqrt((eff_sig_error*numpy.sqrt(luminosity)*cross_section_sig/numpy.sqrt(eff_bkg*cross_section_bkg))**2+(cross_section_sig_error*numpy.sqrt(luminosity)*eff_sig/numpy.sqrt(eff_bkg*cross_section_bkg))**2+(eff_bkg_error*numpy.sqrt(luminosity)*eff_sig*cross_section_sig*cross_section_bkg/(-2*(eff_bkg*cross_section_bkg)**(3/2)))**2+(cross_section_bkg_error*numpy.sqrt(luminosity)*eff_sig*cross_section_sig*eff_bkg/(-2*(eff_bkg*cross_section_bkg)**(3/2)))**2)

### cross sections in pb
higgs_100GeV_cross_section = 0.8432*10**(-12)
higgs_110GeV_cross_section = 0.7808*10**(-12)
higgs_125GeV_cross_section = 0.7002*10**(-12)
higgs_140GeV_cross_section = 0.6351*10**(-12)
z_cross_section = 1418*10**(-12)

higgs_100GeV_cross_section_error = 0.0007517*10**(-12)
higgs_110GeV_cross_section_error = 0.0008196*10**(-12)
higgs_125GeV_cross_section_error = 0.0006399*10**(-12)
higgs_140GeV_cross_section_error = 0.0007003*10**(-12)
z_cross_section_error = 1.431*10**(-12)
### luminosity in inverse fb
luminosity = 100*10**(15)


if signal =="100GeV":
    eff_sig_nn = efficiency(histditaumassnn100GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn = efficiency(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit = efficiency(histditaumasssvfit100GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit = efficiency(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn = signal_to_background(eff_sig_nn,higgs_100GeV_cross_section,eff_bkg_nn,z_cross_section,luminosity)
    sig_to_bkg_svfit = signal_to_background(eff_sig_svfit,higgs_100GeV_cross_section,eff_bkg_svfit,z_cross_section,luminosity)
    eff_sig_nn_error = efficiency_error(histditaumassnn100GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn_error = efficiency_error(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit_error = efficiency_error(histditaumasssvfit100GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit_error = efficiency_error(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn_error = signal_to_background_error(eff_sig_nn,eff_sig_nn_error,higgs_100GeV_cross_section,higgs_100GeV_cross_section_error,eff_bkg_nn,eff_bkg_nn_error,z_cross_section,z_cross_section_error,luminosity)
    sig_to_bkg_svfit_error = signal_to_background_error(eff_sig_svfit,eff_sig_svfit_error,higgs_100GeV_cross_section,higgs_100GeV_cross_section_error,eff_bkg_svfit,eff_bkg_svfit_error,z_cross_section,z_cross_section_error,luminosity)
if signal =="110GeV":
    eff_sig_nn = efficiency(histditaumassnn110GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn = efficiency(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit = efficiency(histditaumasssvfit110GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit = efficiency(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn = signal_to_background(eff_sig_nn,higgs_110GeV_cross_section,eff_bkg_nn,z_cross_section,luminosity)
    sig_to_bkg_svfit = signal_to_background(eff_sig_svfit,higgs_110GeV_cross_section,eff_bkg_svfit,z_cross_section,luminosity)
    eff_sig_nn_error = efficiency_error(histditaumassnn110GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn_error = efficiency_error(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit_error = efficiency_error(histditaumasssvfit110GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit_error = efficiency_error(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn_error = signal_to_background_error(eff_sig_nn,eff_sig_nn_error,higgs_110GeV_cross_section,higgs_110GeV_cross_section_error,eff_bkg_nn,eff_bkg_nn_error,z_cross_section,z_cross_section_error,luminosity)
    sig_to_bkg_svfit_error = signal_to_background_error(eff_sig_svfit,eff_sig_svfit_error,higgs_110GeV_cross_section,higgs_110GeV_cross_section_error,eff_bkg_svfit,eff_bkg_svfit_error,z_cross_section,z_cross_section_error,luminosity)
if signal =="125GeV":
    eff_sig_nn = efficiency(histditaumassnn125GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn = efficiency(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit = efficiency(histditaumasssvfit125GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit = efficiency(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn = signal_to_background(eff_sig_nn,higgs_125GeV_cross_section,eff_bkg_nn,z_cross_section,luminosity)
    sig_to_bkg_svfit = signal_to_background(eff_sig_svfit,higgs_125GeV_cross_section,eff_bkg_svfit,z_cross_section,luminosity)
    eff_sig_nn_error = efficiency_error(histditaumassnn125GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn_error = efficiency_error(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit_error = efficiency_error(histditaumasssvfit125GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit_error = efficiency_error(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn_error = signal_to_background_error(eff_sig_nn,eff_sig_nn_error,higgs_125GeV_cross_section,higgs_125GeV_cross_section_error,eff_bkg_nn,eff_bkg_nn_error,z_cross_section,z_cross_section_error,luminosity)
    sig_to_bkg_svfit_error = signal_to_background_error(eff_sig_svfit,eff_sig_svfit_error,higgs_125GeV_cross_section,higgs_125GeV_cross_section_error,eff_bkg_svfit,eff_bkg_svfit_error,z_cross_section,z_cross_section_error,luminosity)
if signal =="140GeV":
    eff_sig_nn = efficiency(histditaumassnn140GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn = efficiency(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit = efficiency(histditaumasssvfit140GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit = efficiency(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn = signal_to_background(eff_sig_nn,higgs_140GeV_cross_section,eff_bkg_nn,z_cross_section,luminosity)
    sig_to_bkg_svfit = signal_to_background(eff_sig_svfit,higgs_140GeV_cross_section,eff_bkg_svfit,z_cross_section,luminosity)
    eff_sig_nn_error = efficiency_error(histditaumassnn140GeVcomp,signal_left,signal_right,12000)
    eff_bkg_nn_error = efficiency_error(histditaumassnndycomp,signal_left,signal_right,12000)
    eff_sig_svfit_error = efficiency_error(histditaumasssvfit140GeVcomp,signal_left,signal_right,12000)
    eff_bkg_svfit_error = efficiency_error(histditaumasssvfitdycomp,signal_left,signal_right,12000)
    sig_to_bkg_nn_error = signal_to_background_error(eff_sig_nn,eff_sig_nn_error,higgs_140GeV_cross_section,higgs_140GeV_cross_section_error,eff_bkg_nn,eff_bkg_nn_error,z_cross_section,z_cross_section_error,luminosity)
    sig_to_bkg_svfit_error = signal_to_background_error(eff_sig_svfit,eff_sig_svfit_error,higgs_140GeV_cross_section,higgs_140GeV_cross_section_error,eff_bkg_svfit,eff_bkg_svfit_error,z_cross_section,z_cross_section_error,luminosity)

efficiency_comparison = efficiency_comp(eff_sig_nn,eff_bkg_nn,eff_sig_svfit,eff_bkg_svfit)

print "signal:",signal
print "signalrange limits:",signal_left,signal_right
print "NN Signal over sqrt Background with cross_section and luminosity:",sig_to_bkg_nn,"+-",sig_to_bkg_nn_error
print "SVfit Signal over sqrt Background with cross_section and luminosity:",sig_to_bkg_svfit,"+-",sig_to_bkg_svfit_error
print "efficiency comparison:",efficiency_comparison

##########################   save  histograms       ##########################
canv2 = ROOT.TCanvas("di-tau mass using NN and SVfit")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum())
histditaumass.SetMaximum(max_bin*1.08)
histditaumass.Draw("HIST")
histditaumassnn.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.13,0.62,0.35,0.87)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P SAME")
unit_line = ROOT.TLine(50.0,1.0,350.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_hist_name = "%s.png" %(output_name)
canv2.SaveAs(output_hist_name)


canv3 = ROOT.TCanvas("ditaumassnncorr")
histditaumassnncorr.Draw()
line = ROOT.TLine(50.0,50.0,400.0,400.0)
line.Draw("SAME")
line.SetLineWidth(2)
output_hist_corr_name = "%s_corr.png" %(output_name)
canv3.SaveAs(output_hist_corr_name)

canv4 = ROOT.TCanvas("resolution")
histditaumassnnres.Draw()
output_hist_res_name = "%s_res.png" %(output_name)
canv4.SaveAs(output_hist_res_name)

canv5 = ROOT.TCanvas("ditaumass NN correlation with resolution")
histditaumassnncorrres.Draw()
output_hist_corrres_name = "%s_corrres.png" %(output_name)
canv5.SaveAs(output_hist_corrres_name)

canv_use1 = ROOT.TCanvas("nn resolution use")
histditaumassnnrescomp.Draw()
ROOT.gPad.Update()
nn_statbox = histditaumassnnrescomp.FindObject("stats")
nn_color = histditaumassnnrescomp.GetLineColor()
nn_statbox.SetTextColor(1)
nn_statbox.SetLineColor(nn_color)
nn_statbox.SetOptStat(1101)
X1 = nn_statbox.GetX1NDC()
Y1 = nn_statbox.GetY1NDC()
X2 = nn_statbox.GetX2NDC()
Y2 = nn_statbox.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution use")
histditaumasssvfitres.Draw()
ROOT.gPad.Update()
svfit_statbox = histditaumasssvfitres.FindObject("stats")
svfit_color = histditaumasssvfitres.GetLineColor()
svfit_statbox.SetTextColor(1)
svfit_statbox.SetLineColor(svfit_color)
svfit_statbox.SetOptStat(1101)
svfit_statbox.SetX1NDC(X1)
svfit_statbox.SetX2NDC(X2)
svfit_statbox.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox.SetY2NDC(Y1)

canv7 = ROOT.TCanvas("resolution comparison")
max_bin = max(histditaumassnnrescomp.GetMaximum(),histditaumasssvfitres.GetMaximum())
histditaumassnnrescomp.SetMaximum(max_bin*1.08)
histditaumassnnrescomp.Draw()
histditaumasssvfitres.Draw("SAMES")
nn_statbox.Draw("SAME")
svfit_statbox.Draw("SAME")
output_res_compare_name = "%s_rescompar.png" %(output_name)
canv7.SaveAs(output_res_compare_name)

canv9 = ROOT.TCanvas("ditaumass profile resolution comparison")
max_bin9 = max(histprofditaumassnncorrres.GetMaximum(),histprofditaumasssvfitcorrres.GetMaximum())
histprofditaumassnncorrres.SetMaximum(max_bin9*1.8)
histprofditaumassnncorrres.Draw()
histprofditaumasssvfitcorrres.Draw("SAME")
leg9 = ROOT.TLegend(0.13,0.77,0.4,0.87)
leg9.AddEntry(histprofditaumassnncorrres,"Neural Network","PL")
leg9.AddEntry(histprofditaumasssvfitcorrres,"SVfit","PL")
leg9.SetTextSize(0.04)
leg9.Draw()
output_profrescomp_name = "%s_profrescomp.png" %(output_name)
canv9.SaveAs(output_profrescomp_name)

canv11 = ROOT.TCanvas("ditaumass profile abs(resolution) comparisson")
max_bin11 = max(histprofditaumassnncorrabsres.GetMaximum(),histprofditaumasssvfitcorrabsres.GetMaximum())
histprofditaumassnncorrabsres.SetMaximum(max_bin11*1.08)
histprofditaumassnncorrabsres.Draw()
histprofditaumasssvfitcorrabsres.Draw("SAME")
leg11 = ROOT.TLegend(0.13,0.77,0.4,0.87)
leg11.AddEntry(histprofditaumassnncorrabsres,"Neural Network","PL")
leg11.AddEntry(histprofditaumasssvfitcorrabsres,"SVfit","PL")
leg11.SetTextSize(0.04)
leg11.Draw()
output_profabsrescomp_name = "%s_profabsrescomp.png" %(output_name)
canv11.SaveAs(output_profabsrescomp_name)

canv12 = ROOT.TCanvas("ditaumass rms")
histditaumassnnrms.Draw("P")
output_rms_name = "%s_rms.png" %(output_name)
canv12.SaveAs(output_rms_name)


canv13 = ROOT.TCanvas("di-tau mass using NN and SVfit including vismass")
pad1 = ROOT.TPad("pad1","large pad",0.0,0.32,1.0,1.0)
pad2 = ROOT.TPad("pad2","small pad",0.0,0.0,1.0,0.3)
pad1.SetMargin(0.09,0.02,0.02,0.1)
pad2.SetMargin(0.09,0.02,0.35,0.03)
pad1.Draw()
pad2.Draw()
pad1.cd()
max_bin = max(histditaumass.GetMaximum(),histditaumassnn.GetMaximum(),histditaumasssvfit.GetMaximum(),histditauvismass.GetMaximum())
histditaumass.SetMaximum(max_bin*1.3)
histditaumass.Draw("HIST")
histditauvismass.Draw("HIST SAME")
histditaumassnn.Draw("HIST SAME")
histditaumasssvfit.Draw("HIST SAME")
leg2 = ROOT.TLegend(0.67,0.6,0.97,0.87)
leg2.AddEntry(histditaumass,"di-#tau_{gen} mass","PL")
leg2.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg2.AddEntry(histditaumassnn,"di-#tau_{NN} mass","PL")
leg2.AddEntry(histditaumasssvfit,"di-#tau_{SVfit} mass","PL")
leg2.SetTextSize(0.05)
leg2.Draw()
pad2.cd()
histditaumassnnratio.Draw("P")
histditaumasssvfitratio.Draw("P SAME")
unit_line = ROOT.TLine(50.0,1.0,350.0,1.0)
unit_line.SetLineColor(2)
unit_line.Draw("SAME")
output_allhist_name = "%s_all.png" %(output_name)
canv13.SaveAs(output_allhist_name)


canv = ROOT.TCanvas("ditauvismass and ditaumass")
max_bin = max(histditaumassgen.GetMaximum(),histditauvismass.GetMaximum())
histditaumassgen.SetMaximum(max_bin*1.3)
histditaumassgen.Draw()
histditauvismass.Draw("SAME")
leg = ROOT.TLegend(0.55,0.65,0.87,0.87)
leg.AddEntry(histditaumassgen,"di-#tau_{gen} mass","PL")
leg.AddEntry(histditauvismass,"visible di-#tau_{gen} mass","PL")
leg.Draw()
output_massandvismass_name = "%s_massandvismass.png" % (output_name)
canv.SaveAs(output_massandvismass_name)

##### 100 GeV histograms
canv = ROOT.TCanvas("di-tau mass 100GeV using NN and SVfit")
max_bin = max(histditaumass100GeV.GetMaximum(),histditaumassnn100GeV.GetMaximum(),histditaumasssvfit100GeV.GetMaximum())
histditaumassnn100GeV.SetMaximum(max_bin*1.08)
histditaumassnn100GeV.Draw()
histditaumass100GeV.Draw("SAME")
histditaumasssvfit100GeV.Draw("SAME")
leg = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg.AddEntry(histditaumass100GeV,"di-#tau_{gen} mass","PL")
leg.AddEntry(histditaumassnn100GeV,"di-#tau_{NN} mass","PL")
leg.AddEntry(histditaumasssvfit100GeV,"di-#tau_{SVfit} mass","PL")
leg.SetTextSize(0.04)
leg.Draw()
#canv.SetLogy()
output_hist_name = "%s_100GeV.png" %(output_name)
canv.SaveAs(output_hist_name)

canv_use1 = ROOT.TCanvas("nn resolution 100GeV use")
histditaumassnnres100GeV.Draw()
ROOT.gPad.Update()
nn_statbox100GeV = histditaumassnnres100GeV.FindObject("stats")
nn_color = histditaumassnnres100GeV.GetLineColor()
nn_statbox100GeV.SetTextColor(1)
nn_statbox100GeV.SetLineColor(nn_color)
nn_statbox100GeV.SetOptStat(1101)
X1 = nn_statbox100GeV.GetX1NDC()
Y1 = nn_statbox100GeV.GetY1NDC()
X2 = nn_statbox100GeV.GetX2NDC()
Y2 = nn_statbox100GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 100GeV use")
histditaumasssvfitres100GeV.Draw()
ROOT.gPad.Update()
svfit_statbox100GeV = histditaumasssvfitres100GeV.FindObject("stats")
svfit_color = histditaumasssvfitres100GeV.GetLineColor()
svfit_statbox100GeV.SetTextColor(1)
svfit_statbox100GeV.SetLineColor(svfit_color)
svfit_statbox100GeV.SetOptStat(1101)
svfit_statbox100GeV.SetX1NDC(X1)
svfit_statbox100GeV.SetX2NDC(X2)
svfit_statbox100GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox100GeV.SetY2NDC(Y1)

canv = ROOT.TCanvas("resolution comparison 100GeV")
max_bin = max(histditaumassnnres100GeV.GetMaximum(),histditaumasssvfitres100GeV.GetMaximum())
histditaumassnnres100GeV.SetMaximum(max_bin*1.08)
histditaumassnnres100GeV.Draw()
histditaumasssvfitres100GeV.Draw("SAMES")
nn_statbox100GeV.Draw("SAME")
svfit_statbox100GeV.Draw("SAME")
output_res_compare_name = "%s_100GeV_rescompar.png" %(output_name)
canv.SaveAs(output_res_compare_name)

##### 110 GeV histograms
canv = ROOT.TCanvas("di-tau mass 110GeV using NN and SVfit")
max_bin = max(histditaumass110GeV.GetMaximum(),histditaumassnn110GeV.GetMaximum(),histditaumasssvfit110GeV.GetMaximum())
histditaumassnn110GeV.SetMaximum(max_bin*1.08)
histditaumassnn110GeV.Draw()
histditaumass110GeV.Draw("SAME")
histditaumasssvfit110GeV.Draw("SAME")
leg = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg.AddEntry(histditaumass110GeV,"di-#tau_{gen} mass","PL")
leg.AddEntry(histditaumassnn110GeV,"di-#tau_{NN} mass","PL")
leg.AddEntry(histditaumasssvfit110GeV,"di-#tau_{SVfit} mass","PL")
leg.SetTextSize(0.04)
leg.Draw()
#canv.SetLogy()
output_hist_name = "%s_110GeV.png" %(output_name)
canv.SaveAs(output_hist_name)

canv_use1 = ROOT.TCanvas("nn resolution 110GeV use")
histditaumassnnres110GeV.Draw()
ROOT.gPad.Update()
nn_statbox110GeV = histditaumassnnres110GeV.FindObject("stats")
nn_color = histditaumassnnres110GeV.GetLineColor()
nn_statbox110GeV.SetTextColor(1)
nn_statbox110GeV.SetLineColor(nn_color)
nn_statbox110GeV.SetOptStat(1101)
X1 = nn_statbox110GeV.GetX1NDC()
Y1 = nn_statbox110GeV.GetY1NDC()
X2 = nn_statbox110GeV.GetX2NDC()
Y2 = nn_statbox110GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 110GeV use")
histditaumasssvfitres110GeV.Draw()
ROOT.gPad.Update()
svfit_statbox110GeV = histditaumasssvfitres110GeV.FindObject("stats")
svfit_color = histditaumasssvfitres110GeV.GetLineColor()
svfit_statbox110GeV.SetTextColor(1)
svfit_statbox110GeV.SetLineColor(svfit_color)
svfit_statbox110GeV.SetOptStat(1101)
svfit_statbox110GeV.SetX1NDC(X1)
svfit_statbox110GeV.SetX2NDC(X2)
svfit_statbox110GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox110GeV.SetY2NDC(Y1)

canv = ROOT.TCanvas("resolution comparison 110GeV")
max_bin = max(histditaumassnnres110GeV.GetMaximum(),histditaumasssvfitres110GeV.GetMaximum())
histditaumassnnres110GeV.SetMaximum(max_bin*1.08)
histditaumassnnres110GeV.Draw()
histditaumasssvfitres110GeV.Draw("SAMES")
nn_statbox110GeV.Draw("SAME")
svfit_statbox110GeV.Draw("SAME")
output_res_compare_name = "%s_110GeV_rescompar.png" %(output_name)
canv.SaveAs(output_res_compare_name)

##### 125 GeV histograms
canv = ROOT.TCanvas("di-tau mass 125GeV using NN and SVfit")
max_bin = max(histditaumass125GeV.GetMaximum(),histditaumassnn125GeV.GetMaximum(),histditaumasssvfit125GeV.GetMaximum())
histditaumassnn125GeV.SetMaximum(max_bin*1.08)
histditaumassnn125GeV.Draw()
histditaumass125GeV.Draw("SAME")
histditaumasssvfit125GeV.Draw("SAME")
leg14 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg14.AddEntry(histditaumass125GeV,"di-#tau_{gen} mass","PL")
leg14.AddEntry(histditaumassnn125GeV,"di-#tau_{NN} mass","PL")
leg14.AddEntry(histditaumasssvfit125GeV,"di-#tau_{SVfit} mass","PL")
leg14.SetTextSize(0.04)
leg14.Draw()
#canv.SetLogy()
output_hist_name = "%s_125GeV.png" %(output_name)
canv.SaveAs(output_hist_name)

canv_use1 = ROOT.TCanvas("nn resolution 125GeV use")
histditaumassnnres125GeV.Draw()
ROOT.gPad.Update()
nn_statbox125GeV = histditaumassnnres125GeV.FindObject("stats")
nn_color = histditaumassnnres125GeV.GetLineColor()
nn_statbox125GeV.SetTextColor(1)
nn_statbox125GeV.SetLineColor(nn_color)
nn_statbox125GeV.SetOptStat(1101)
X1 = nn_statbox125GeV.GetX1NDC()
Y1 = nn_statbox125GeV.GetY1NDC()
X2 = nn_statbox125GeV.GetX2NDC()
Y2 = nn_statbox125GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 125GeV use")
histditaumasssvfitres125GeV.Draw()
ROOT.gPad.Update()
svfit_statbox125GeV = histditaumasssvfitres125GeV.FindObject("stats")
svfit_color = histditaumasssvfitres125GeV.GetLineColor()
svfit_statbox125GeV.SetTextColor(1)
svfit_statbox125GeV.SetLineColor(svfit_color)
svfit_statbox125GeV.SetOptStat(1101)
svfit_statbox125GeV.SetX1NDC(X1)
svfit_statbox125GeV.SetX2NDC(X2)
svfit_statbox125GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox125GeV.SetY2NDC(Y1)

canv15 = ROOT.TCanvas("resolution comparison 125GeV")
max_bin15 = max(histditaumassnnres125GeV.GetMaximum(),histditaumasssvfitres125GeV.GetMaximum())
histditaumassnnres125GeV.SetMaximum(max_bin15*1.08)
histditaumassnnres125GeV.Draw()
histditaumasssvfitres125GeV.Draw("SAMES")
nn_statbox125GeV.Draw("SAME")
svfit_statbox125GeV.Draw("SAME")
output_res125GeV_compare_name = "%s_125GeV_rescompar.png" %(output_name)
canv15.SaveAs(output_res125GeV_compare_name)

##### 140 GeV histograms
canv = ROOT.TCanvas("di-tau mass 140GeV using NN and SVfit")
max_bin = max(histditaumass140GeV.GetMaximum(),histditaumassnn140GeV.GetMaximum(),histditaumasssvfit140GeV.GetMaximum())
histditaumassnn140GeV.SetMaximum(max_bin*1.08)
histditaumassnn140GeV.Draw()
histditaumass140GeV.Draw("SAME")
histditaumasssvfit140GeV.Draw("SAME")
leg = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg.AddEntry(histditaumass140GeV,"di-#tau_{gen} mass","PL")
leg.AddEntry(histditaumassnn140GeV,"di-#tau_{NN} mass","PL")
leg.AddEntry(histditaumasssvfit140GeV,"di-#tau_{SVfit} mass","PL")
leg.SetTextSize(0.04)
leg.Draw()
#canv.SetLogy()
output_hist_name = "%s_140GeV.png" %(output_name)
canv.SaveAs(output_hist_name)


canv_use1 = ROOT.TCanvas("nn resolution 140GeV use")
histditaumassnnres140GeV.Draw()
ROOT.gPad.Update()
nn_statbox140GeV = histditaumassnnres140GeV.FindObject("stats")
nn_color = histditaumassnnres140GeV.GetLineColor()
nn_statbox140GeV.SetTextColor(1)
nn_statbox140GeV.SetLineColor(nn_color)
nn_statbox140GeV.SetOptStat(1101)
X1 = nn_statbox140GeV.GetX1NDC()
Y1 = nn_statbox140GeV.GetY1NDC()
X2 = nn_statbox140GeV.GetX2NDC()
Y2 = nn_statbox140GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 140GeV use")
histditaumasssvfitres140GeV.Draw()
ROOT.gPad.Update()
svfit_statbox140GeV = histditaumasssvfitres140GeV.FindObject("stats")
svfit_color = histditaumasssvfitres140GeV.GetLineColor()
svfit_statbox140GeV.SetTextColor(1)
svfit_statbox140GeV.SetLineColor(svfit_color)
svfit_statbox140GeV.SetOptStat(1101)
svfit_statbox140GeV.SetX1NDC(X1)
svfit_statbox140GeV.SetX2NDC(X2)
svfit_statbox140GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox140GeV.SetY2NDC(Y1)

canv = ROOT.TCanvas("resolution comparison 140GeV")
max_bin = max(histditaumassnnres140GeV.GetMaximum(),histditaumasssvfitres140GeV.GetMaximum())
histditaumassnnres140GeV.SetMaximum(max_bin*1.08)
histditaumassnnres140GeV.Draw()
histditaumasssvfitres140GeV.Draw("SAMES")
nn_statbox140GeV.Draw("SAME")
svfit_statbox140GeV.Draw("SAME")
output_res_compare_name = "%s_140GeV_rescompar.png" %(output_name)
canv.SaveAs(output_res_compare_name)

##### DY histograms
canv26 = ROOT.TCanvas("di-tau mass DY using NN and SVfit")
max_bindy = max(histditaumassdy.GetMaximum(),histditaumassnndy.GetMaximum(),histditaumasssvfitdy.GetMaximum())
histditaumassnndy.SetMaximum(max_bindy*1.08)
histditaumassnndy.Draw()
histditaumassdy.Draw("SAME")
histditaumasssvfitdy.Draw("SAME")
leg26 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg26.AddEntry(histditaumassdy,"di-#tau_{gen} mass","PL")
leg26.AddEntry(histditaumassnndy,"di-#tau_{NN} mass","PL")
leg26.AddEntry(histditaumasssvfitdy,"di-#tau_{SVfit} mass","PL")
leg26.SetTextSize(0.04)
leg26.Draw()
#canv26.SetLogy()
output_histdy_name = "%s_dy.png" %(output_name)
canv26.SaveAs(output_histdy_name)

canv_use1 = ROOT.TCanvas("nn resolution DY use")
histditaumassnnresdy.Draw()
ROOT.gPad.Update()
nn_statboxdy = histditaumassnnresdy.FindObject("stats")
nn_color = histditaumassnnresdy.GetLineColor()
nn_statboxdy.SetTextColor(1)
nn_statboxdy.SetLineColor(nn_color)
nn_statboxdy.SetOptStat(1101)
X1 = nn_statboxdy.GetX1NDC()
Y1 = nn_statboxdy.GetY1NDC()
X2 = nn_statboxdy.GetX2NDC()
Y2 = nn_statboxdy.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution DY use")
histditaumasssvfitresdy.Draw()
ROOT.gPad.Update()
svfit_statboxdy = histditaumasssvfitresdy.FindObject("stats")
svfit_color = histditaumasssvfitresdy.GetLineColor()
svfit_statboxdy.SetTextColor(1)
svfit_statboxdy.SetLineColor(svfit_color)
svfit_statboxdy.SetOptStat(1101)
svfit_statboxdy.SetX1NDC(X1)
svfit_statboxdy.SetX2NDC(X2)
svfit_statboxdy.SetY1NDC(Y1-(Y2-Y1))
svfit_statboxdy.SetY2NDC(Y1)

canv27 = ROOT.TCanvas("resolution comparison DY")
max_bin27 = max(histditaumassnnresdy.GetMaximum(),histditaumasssvfitresdy.GetMaximum())
histditaumassnnresdy.SetMaximum(max_bin27*1.08)
histditaumassnnresdy.Draw()
histditaumasssvfitresdy.Draw("SAMES")
nn_statboxdy.Draw("SAME")
svfit_statboxdy.Draw("SAME")
output_resdy_compare_name = "%s_dy_rescompar.png" %(output_name)
canv27.SaveAs(output_resdy_compare_name)

##### 250 GeV histograms
canv16 = ROOT.TCanvas("di-tau mass 250GeV using NN and SVfit")
max_bin250GeV = max(histditaumass250GeV.GetMaximum(),histditaumassnn250GeV.GetMaximum(),histditaumasssvfit250GeV.GetMaximum())
histditaumassnn250GeV.SetMaximum(max_bin250GeV*1.08)
histditaumassnn250GeV.Draw()
histditaumass250GeV.Draw("SAME")
histditaumasssvfit250GeV.Draw("SAME")
leg16 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg16.AddEntry(histditaumass250GeV,"di-#tau_{gen} mass","PL")
leg16.AddEntry(histditaumassnn250GeV,"di-#tau_{NN} mass","PL")
leg16.AddEntry(histditaumasssvfit250GeV,"di-#tau_{SVfit} mass","PL")
leg16.SetTextSize(0.04)
leg16.Draw()
#canv16.SetLogy()
output_hist250GeV_name = "%s_250GeV.png" %(output_name)
canv16.SaveAs(output_hist250GeV_name)

canv_use1 = ROOT.TCanvas("nn resolution 250GeV use")
histditaumassnnres250GeV.Draw()
ROOT.gPad.Update()
nn_statbox250GeV = histditaumassnnres250GeV.FindObject("stats")
nn_color = histditaumassnnres250GeV.GetLineColor()
nn_statbox250GeV.SetTextColor(1)
nn_statbox250GeV.SetLineColor(nn_color)
nn_statbox250GeV.SetOptStat(1101)
X1 = nn_statbox250GeV.GetX1NDC()
Y1 = nn_statbox250GeV.GetY1NDC()
X2 = nn_statbox250GeV.GetX2NDC()
Y2 = nn_statbox250GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 250GeV use")
histditaumasssvfitres250GeV.Draw()
ROOT.gPad.Update()
svfit_statbox250GeV = histditaumasssvfitres250GeV.FindObject("stats")
svfit_color = histditaumasssvfitres250GeV.GetLineColor()
svfit_statbox250GeV.SetTextColor(1)
svfit_statbox250GeV.SetLineColor(svfit_color)
svfit_statbox250GeV.SetOptStat(1101)
svfit_statbox250GeV.SetX1NDC(X1)
svfit_statbox250GeV.SetX2NDC(X2)
svfit_statbox250GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox250GeV.SetY2NDC(Y1)

canv17 = ROOT.TCanvas("resolution comparison 250GeV")
max_bin17 = max(histditaumassnnres250GeV.GetMaximum(),histditaumasssvfitres250GeV.GetMaximum())
histditaumassnnres250GeV.SetMaximum(max_bin17*1.08)
histditaumassnnres250GeV.Draw()
histditaumasssvfitres250GeV.Draw("SAMES")
nn_statbox250GeV.Draw("SAME")
svfit_statbox250GeV.Draw("SAME")
output_res250GeV_compare_name = "%s_250GeV_rescompar.png" %(output_name)
canv17.SaveAs(output_res250GeV_compare_name)

##### 180 GeV histograms
canv18 = ROOT.TCanvas("di-tau mass 180GeV using NN and SVfit")
max_bin180GeV = max(histditaumass180GeV.GetMaximum(),histditaumassnn180GeV.GetMaximum(),histditaumasssvfit180GeV.GetMaximum())
histditaumassnn180GeV.SetMaximum(max_bin180GeV*1.08)
histditaumassnn180GeV.Draw()
histditaumass180GeV.Draw("SAME")
histditaumasssvfit180GeV.Draw("SAME")
leg18 = ROOT.TLegend(0.13,0.65,0.4,0.87)
leg18.AddEntry(histditaumass180GeV,"di-#tau_{gen} mass","PL")
leg18.AddEntry(histditaumassnn180GeV,"di-#tau_{NN} mass","PL")
leg18.AddEntry(histditaumasssvfit180GeV,"di-#tau_{SVfit} mass","PL")
leg18.SetTextSize(0.04)
leg18.Draw()
#canv18.SetLogy()
output_hist180GeV_name = "%s_180GeV.png" %(output_name)
canv18.SaveAs(output_hist180GeV_name)

canv_use1 = ROOT.TCanvas("nn resolution 180GeV use")
histditaumassnnres180GeV.Draw()
ROOT.gPad.Update()
nn_statbox180GeV = histditaumassnnres180GeV.FindObject("stats")
nn_color = histditaumassnnres180GeV.GetLineColor()
nn_statbox180GeV.SetTextColor(1)
nn_statbox180GeV.SetLineColor(nn_color)
nn_statbox180GeV.SetOptStat(1101)
X1 = nn_statbox180GeV.GetX1NDC()
Y1 = nn_statbox180GeV.GetY1NDC()
X2 = nn_statbox180GeV.GetX2NDC()
Y2 = nn_statbox180GeV.GetY2NDC()

canv_use2 = ROOT.TCanvas("svfit resolution 180GeV use")
histditaumasssvfitres180GeV.Draw()
ROOT.gPad.Update()
svfit_statbox180GeV = histditaumasssvfitres180GeV.FindObject("stats")
svfit_color = histditaumasssvfitres180GeV.GetLineColor()
svfit_statbox180GeV.SetTextColor(1)
svfit_statbox180GeV.SetLineColor(svfit_color)
svfit_statbox180GeV.SetOptStat(1101)
svfit_statbox180GeV.SetX1NDC(X1)
svfit_statbox180GeV.SetX2NDC(X2)
svfit_statbox180GeV.SetY1NDC(Y1-(Y2-Y1))
svfit_statbox180GeV.SetY2NDC(Y1)

canv19 = ROOT.TCanvas("resolution comparison 180GeV")
max_bin19 = max(histditaumassnnres180GeV.GetMaximum(),histditaumasssvfitres180GeV.GetMaximum())
histditaumassnnres180GeV.SetMaximum(max_bin19*1.08)
histditaumassnnres180GeV.Draw()
histditaumasssvfitres180GeV.Draw("SAMES")
nn_statbox180GeV.Draw("SAME")
svfit_statbox180GeV.Draw("SAME")
output_res180GeV_compare_name = "%s_180GeV_rescompar.png" %(output_name)
canv19.SaveAs(output_res180GeV_compare_name)

canv21 = ROOT.TCanvas("ditaumasssvfitcorr")
histditaumasssvfitcorr.Draw()
line = ROOT.TLine(0.0,0.0,400.0,400.0)
line.Draw("SAME")
line.SetLineWidth(2)
output_hist_svfit_corr_name = "%s_svfit_corr.png" %(output_name)
canv21.SaveAs(output_hist_svfit_corr_name)

canv22 = ROOT.TCanvas("ditaumass SVfit correlation with resolution")
histditaumasssvfitcorrres.Draw()
output_hist_svfit_corrres_name = "%s_svfit_corrres.png" %(output_name)
canv22.SaveAs(output_hist_svfit_corrres_name)


if bias_correction == "gen" or bias_correction == "reco":
    canv = ROOT.TCanvas("ditaumass profile resolution comparison before")
    profditaumassnncorrresbefore.Draw()
    profditaumasssvfitcorrresbefore.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.77,0.4,0.87)
    leg.AddEntry(profditaumassnncorrresbefore,"Neural Network","PL")
    leg.AddEntry(profditaumasssvfitcorrresbefore,"SVfit","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    output_profrescompbefore_name = "%s_profrescompbefore.png" %(output_name)
    canv.SaveAs(output_profrescompbefore_name)

#### Higgs and DY histograms
if signal == "100GeV" and signalrange == "no" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass100GeVcomp.GetMaximum()
    histditaumass100GeVcomp.SetMaximum(max_bin*0.14)
    histditaumass100GeVcomp.Draw()
    histditaumassdycomp.Draw("SAME")
    histditaumassnn100GeVcomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit100GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    leg.AddEntry(histditaumass100GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn100GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit100GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    #canv.SetLogy()
    output_hist100GeVDY_name = "%s_100GeVandDY.png" %(output_name)
    canv.SaveAs(output_hist100GeVDY_name)

if signal == "100GeV" and signalrange == "yes" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass100GeVcomp.GetMaximum()
    histditaumass100GeVcomp.SetMaximum(max_bin*0.05)
    histditaumass100GeVcomp.Draw()
    histditaumassdycomp.Draw("SAME")
    histditaumassnn100GeVcomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit100GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    leg.AddEntry(histditaumass100GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn100GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit100GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    output_hist100GeVDY_name = "%s_100GeVandDY_signalrange.png" %(output_name)
    canv.SaveAs(output_hist100GeVDY_name)

if signal == "110GeV" and signalrange == "no" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass110GeVcomp.GetMaximum()
    histditaumass110GeVcomp.SetMaximum(max_bin*0.37)
    #histditaumassnn110GeVcomp.SetMaximum(max_bin*10.0)
    histditaumass110GeVcomp.Draw()
    #histditaumassnn110GeVcomp.Draw()
    #histditaumass110GeVcomp.Draw("SAME")
    histditaumassdycomp.Draw("SAME")
    histditaumassnn110GeVcomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit110GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    #leg = ROOT.TLegend(0.36,0.53,0.68,0.87)
    leg.AddEntry(histditaumass110GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn110GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit110GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    #canv.SetLogy()
    output_hist110GeVDY_name = "%s_110GeVandDY.png" %(output_name)
    canv.SaveAs(output_hist110GeVDY_name)

if signal == "110GeV" and signalrange == "yes" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass110GeVcomp.GetMaximum()
    histditaumass110GeVcomp.SetMaximum(max_bin*0.05)
    histditaumass110GeVcomp.Draw()
    histditaumassdycomp.Draw("SAME")
    histditaumassnn110GeVcomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit110GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    leg.AddEntry(histditaumass110GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn110GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit110GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    output_hist110GeVDY_name = "%s_110GeVandDY_signalrange.png" %(output_name)
    canv.SaveAs(output_hist110GeVDY_name)

if signal == "125GeV" and signalrange == "no" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass125GeVcomp.GetMaximum()
    histditaumass125GeVcomp.SetMaximum(max_bin*0.25)
    histditaumass125GeVcomp.Draw()
    histditaumassnn125GeVcomp.Draw("SAME")
    histditaumassdycomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit125GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    #leg = ROOT.TLegend(0.38,0.53,0.7,0.87)
    leg.AddEntry(histditaumass125GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn125GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit125GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    #canv.SetLogy()
    output_hist125GeVDY_name = "%s_125GeVandDY.png" %(output_name)
    canv.SaveAs(output_hist125GeVDY_name)


if signal == "125GeV" and signalrange == "yes" and signalrange_tight == "no":
    canv25 = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass125GeV.GetMaximum()
    histditaumass125GeVcomp.SetMaximum(max_bin*0.05)
    histditaumass125GeVcomp.Draw()
    histditaumassnn125GeVcomp.Draw("SAME")
    histditaumassdycomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit125GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg25 = ROOT.TLegend(0.13,0.53,0.45,0.87)
    leg25.AddEntry(histditaumass125GeVcomp,"di-#tau_{gen} mass S","PL")
    leg25.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg25.AddEntry(histditaumassnn125GeVcomp,"di-#tau_{NN} mass S","PL")
    leg25.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg25.AddEntry(histditaumasssvfit125GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg25.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg25.SetTextSize(0.04)
    leg25.Draw()
    output_hist125GeVDY_name = "%s_125GeVandDY_signalrange.png" %(output_name)
    canv25.SaveAs(output_hist125GeVDY_name)

if signal == "140GeV" and signalrange == "no" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass140GeVcomp.GetMaximum()
    histditaumass140GeVcomp.SetMaximum(max_bin*0.3)
    histditaumass140GeVcomp.Draw()
    histditaumassnn140GeVcomp.Draw("SAME")
    histditaumassdycomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit140GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.55,0.45,0.87)
    #leg = ROOT.TLegend(0.38,0.53,0.7,0.87)
    leg.AddEntry(histditaumass140GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn140GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit140GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    #canv.SetLogy()
    output_hist140GeVDY_name = "%s_140GeVandDY.png" %(output_name)
    canv.SaveAs(output_hist140GeVDY_name)

if signal == "140GeV" and signalrange == "yes" and signalrange_tight == "no":
    canv = ROOT.TCanvas("di-tau mass Signal and DY using NN and SVfit")
    max_bin = histditaumass140GeVcomp.GetMaximum()
    histditaumass140GeVcomp.SetMaximum(max_bin*0.05)
    histditaumass140GeVcomp.Draw()
    histditaumassdycomp.Draw("SAME")
    histditaumassnn140GeVcomp.Draw("SAME")
    histditaumassnndycomp.Draw("SAME")
    histditaumasssvfit140GeVcomp.Draw("SAME")
    histditaumasssvfitdycomp.Draw("SAME")
    leg = ROOT.TLegend(0.13,0.53,0.45,0.87)
    leg.AddEntry(histditaumass140GeVcomp,"di-#tau_{gen} mass S","PL")
    leg.AddEntry(histditaumassdycomp,"di-#tau_{gen} mass B","PL")
    leg.AddEntry(histditaumassnn140GeVcomp,"di-#tau_{NN} mass S","PL")
    leg.AddEntry(histditaumassnndycomp,"di-#tau_{NN} mass B","PL")
    leg.AddEntry(histditaumasssvfit140GeVcomp,"di-#tau_{SVfit} mass S","PL")
    leg.AddEntry(histditaumasssvfitdycomp,"di-#tau_{SVfit} mass B","PL")
    leg.SetTextSize(0.04)
    leg.Draw()
    output_hist140GeVDY_name = "%s_140GeVandDY_signalrange.png" %(output_name)
    canv.SaveAs(output_hist140GeVDY_name)
if signalrange == "yes":
    output_file.close()
