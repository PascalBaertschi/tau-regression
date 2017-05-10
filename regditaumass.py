import numpy
import pandas
import ROOT
import random
seed = 7
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
#from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#myfile = ROOT.TFile("regditaumass.root","RECREATE")
#load dataset housing
dataframe_housing = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset_housing = dataframe_housing.values

#load dataset hadronic decays
dataframe_mass_train = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train = dataframe_mass_train.values
dataframe_mass_test= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test = dataframe_mass_test.values
dataframe_mass= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass = dataframe_mass.values

dataframe_mass_train_total_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total_had = dataframe_mass_train_total_had.values
dataframe_mass_test_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total_had = dataframe_mass_test_total_had.values
dataframe_mass_total_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_total_had = dataframe_mass_total_had.values

dataframe_mass_train_inclnu_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu_had = dataframe_mass_train_inclnu_had.values
dataframe_mass_test_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu_had = dataframe_mass_test_inclnu_had.values
dataframe_mass_inclnu_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_had_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu_had = dataframe_mass_inclnu_had.values


dataframe_vismass_train_had = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_train_had = dataframe_vismass_train_had.values
dataframe_vismass_test_had= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_had_1e6_new.csv",delim_whitespace=False,header=None)
dataset_vismass_test_had = dataframe_vismass_test_had.values


#load dataset all decays
dataframe_mass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_all = dataframe_mass_train_all.values
dataframe_mass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_all = dataframe_mass_test_all.values
dataframe_mass_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_all = dataframe_mass_all.values

dataframe_mass_train_inclnu = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_inclnu = dataframe_mass_train_inclnu.values
dataframe_mass_test_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_inclnu = dataframe_mass_test_inclnu.values
dataframe_mass_inclnu= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_inclnu_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_inclnu = dataframe_mass_inclnu.values

dataframe_mass_train_total = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_train_total = dataframe_mass_train_total.values
dataframe_mass_test_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_test_total = dataframe_mass_test_total.values
dataframe_mass_total= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_mass_total_1e6.csv",delim_whitespace=False,header=None)
dataset_mass_total = dataframe_mass_total.values

dataframe_train_theta = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_train_theta = dataframe_train_theta.values
dataframe_test_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_test_theta = dataframe_test_theta.values
dataframe_theta= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/reg_ditau_theta_1e6.csv",delim_whitespace=False,header=None)
dataset_theta = dataframe_theta.values

dataframe_vismass_train_all = pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/train_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_train_all = dataframe_vismass_train_all.values
dataframe_vismass_test_all= pandas.read_csv("/mnt/t3nfs01/data01/shome/pbaertsc/tauinitial/CMSSW_8_0_25/src/XtautauML/test_reg_ditau_vismass_all_1e6.csv",delim_whitespace=False,header=None)
dataset_vismass_test_all = dataframe_vismass_test_all.values




# split into input and output variables
#hadronic decays
train_input = dataset_mass_train[:,0:7]
train_output = dataset_mass_train[:,7]
test_input = dataset_mass_test[:,0:7]
test_output = dataset_mass_test[:,7]
mass_input = dataset_mass[:,0:7]
mass_output = dataset_mass[:,7]

train_input_inclnu_had = dataset_mass_train_inclnu_had[:,0:8]
train_output_inclnu_had = dataset_mass_train_inclnu_had[:,8]
test_input_inclnu_had = dataset_mass_test_inclnu_had[:,0:8]
test_output_inclnu_had = dataset_mass_test_inclnu_had[:,8]
mass_input_inclnu_had = dataset_mass_inclnu_had[:,0:8]
mass_output_inclnu_had = dataset_mass_inclnu_had[:,8]

train_input_total_had = dataset_mass_train_total_had[:,0:4]
train_output_total_had = dataset_mass_train_total_had[:,4]
test_input_total_had = dataset_mass_test_total_had[:,0:4]
test_output_total_had = dataset_mass_test_total_had[:,4]
mass_input_total_had = dataset_mass_total_had[:,0:4]
mass_output_total_had = dataset_mass_total_had[:,4]

train_input_vismass_had = dataset_vismass_train_had[:,0:4]
train_output_vismass_had = dataset_vismass_train_had[:,4]
test_input_vismass_had = dataset_vismass_test_had[:,0:4]
test_output_vismass_had = dataset_vismass_test_had[:,4]


#all decays
train_input_all = dataset_mass_train_inclnu[:,0:6]
train_output_all = dataset_mass_train_inclnu[:,8]
test_input_all = dataset_mass_test_inclnu[:,0:6]
test_output_all = dataset_mass_test_inclnu[:,8]
mass_input_all = dataset_mass_inclnu[:,0:6]
mass_output_all=dataset_mass_inclnu[:,8]

train_input_inclnu = dataset_mass_train_inclnu[:,0:8]
train_output_inclnu = dataset_mass_train_inclnu[:,8]
test_input_inclnu = dataset_mass_test_inclnu[:,0:8]
test_output_inclnu = dataset_mass_test_inclnu[:,8]
mass_input_inclnu = dataset_mass_inclnu[:,0:8]
mass_output_inclnu = dataset_mass_inclnu[:,8]

train_input_total = dataset_mass_train_total[:,0:4]
train_output_total = dataset_mass_train_total[:,4]
test_input_total = dataset_mass_test_total[:,0:4]
test_output_total = dataset_mass_test_total[:,4]
mass_input_total = dataset_mass_total[:,0:4]
mass_output_total = dataset_mass_total[:,4]
train_length_total = len(train_output_total)
test_length_total = len(test_output_total)


train_input_theta = dataset_train_theta[:,0:4]
train_output_theta = dataset_train_theta[:,4]
test_input_theta = dataset_test_theta[:,0:4]
test_output_theta = dataset_test_theta[:,4]
theta_input = dataset_theta[:,0:4]
theta_output = dataset_theta[:,4]

train_input_vismass_all = dataset_vismass_train_all[:,0:4]
train_output_vismass_all = dataset_vismass_train_all[:,4]
test_input_vismass_all = dataset_vismass_test_all[:,0:4]
test_output_vismass_all = dataset_vismass_test_all[:,4]

train_px = dataset_train_theta[:,0]
train_py = dataset_train_theta[:,1]
train_pz = dataset_train_theta[:,2]
train_E = dataset_train_theta[:,3]
train_theta = dataset_train_theta[:,4]
test_px = dataset_test_theta[:,0]
test_py = dataset_test_theta[:,1]
test_pz = dataset_test_theta[:,2]
test_E = dataset_test_theta[:,3]
test_theta = dataset_test_theta[:,4]


train_length = len(dataset_mass_train_total[:,0])
test_length = len(dataset_mass_test_total[:,0])
housing_length = len(dataset_housing[:,0])
train_length_housing = int(round(housing_length*0.7))


#using the housing data for the play model
"""
train_input_play = dataset_housing[0:train_length_housing,0]
test_input_play = dataset_housing[train_length_housing:housing_length,0]
train_input_play = train_input_play.tolist()
test_input_play = test_input_play.tolist()
train_output_play = []
for i in range(0,len(train_input_play)):
    train_output_play.append(train_input_play[i]*train_input_play[i])
test_output_play = []
for j in range(0,len(test_input_play)):
    test_output_play.append(test_input_play[j]*test_input_play[j])
#overfit_play = train_input_play[0:len(test_input_play)]
"""
#using the mass_total data for the play model
"""
train_length_play = len(dataset_mass_train_total[:,0])
test_length_play = len(dataset_mass_test_total[:,0])

for i in range(0,train_length_play):
    dataset_mass_train_total[i,0] = dataset_mass_train_total[i,0]*dataset_mass_train_total[i,0]
    dataset_mass_train_total[i,1] = dataset_mass_train_total[i,1]*dataset_mass_train_total[i,1]
    dataset_mass_train_total[i,2] = dataset_mass_train_total[i,2]*dataset_mass_train_total[i,2]
    dataset_mass_train_total[i,3] = dataset_mass_train_total[i,3]*dataset_mass_train_total[i,3]
    dataset_mass_train_total[i,4] = dataset_mass_train_total[i,4]*dataset_mass_train_total[i,4]

for i in range(0,test_length_play):
    dataset_mass_test_total[i,0] = dataset_mass_test_total[i,0]*dataset_mass_test_total[i,0]
    dataset_mass_test_total[i,1] = dataset_mass_test_total[i,1]*dataset_mass_test_total[i,1]
    dataset_mass_test_total[i,2] = dataset_mass_test_total[i,2]*dataset_mass_test_total[i,2]
    dataset_mass_test_total[i,3] = dataset_mass_test_total[i,3]*dataset_mass_test_total[i,3]
    dataset_mass_test_total[i,4] = dataset_mass_test_total[i,4]*dataset_mass_test_total[i,4]

train_input_play = dataset_mass_train_total[:,0:4]
train_output_play = dataset_mass_train_total[:,4]
test_input_play = dataset_mass_test_total[:,0:4]
test_output_play = dataset_mass_test_total[:,4]
"""
#using random numbers for the play model
"""
train_input_play = []
for i in range(0,100000):
    train_input_play.append(random.uniform(0,101))
test_input_play = []
for k in range(0,30000):
    test_input_play.append(random.uniform(0,101))

train_length_play = len(train_input_play)
test_length_play = len(test_input_play)

train_output_play = []
test_output_play = []
for i in range(0,train_length_play):
    train_output_play.append(train_input_play[i]*train_input_play[i])
for j in range(0,test_length_play):
    test_output_play.append(test_input_play[j]*test_input_play[j])
"""

#histogram of ditau mass with regression only hadronic decays
histditaumassreg = ROOT.TH1D("ditaumassreg","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassreg.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreg.GetYaxis().SetTitle("number of occurence")
histditaumassreg.SetLineColor(4)
histditaumassreg.SetStats(0)
histditaumassregout = ROOT.TH1D("ditaumassregout","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassregout.SetLineColor(2)
histditaumassregout.SetStats(0)

#histogram of ditau mass with regression all particles
histditaumassregall = ROOT.TH1D("ditaumassregall","reconstructing di-#tau mass using machine learning",100,0,100)
histditaumassregall.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregall.GetYaxis().SetTitle("number of occurence")
histditaumassregall.GetYaxis().SetTitleOffset(1.5)
histditaumassregall.SetLineColor(4)
histditaumassregall.SetStats(0)
histditaumassregallout = ROOT.TH1D("ditaumassregallout","di-#tau mass using regression all decays out",100,0,100)
histditaumassregallout.SetLineColor(2)
histditaumassregallout.SetStats(0)
histditaumassregalloverfit = ROOT.TH1D("ditaumassregalloverfit","disdsdsd",100,0,100)
histditaumassregalloverfit.SetLineColor(3)
histditaumassregalloverfit.SetLineStyle(2)
histditaumassregalloverfit.SetStats(0)

#histogram of ditau mass with regression all particles with all parameter (including pz of neutrino)
histditaumassreginclnu = ROOT.TH1D("ditaumassreginclnu","machine learning example using visible/neutrino 4Vectors",100,0,100)
histditaumassreginclnu.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreginclnu.GetYaxis().SetTitle("number of occurence")
histditaumassreginclnu.GetYaxis().SetTitleOffset(1.5)
histditaumassreginclnu.SetLineColor(4)
histditaumassreginclnu.SetStats(0)
histditaumassreginclnuout = ROOT.TH1D("ditaumassreginclnuout","ssss",100,0,100)
histditaumassreginclnuout.SetLineColor(2)
histditaumassreginclnuout.SetStats(0)


#histogram of ditau mass with regression all particles with all parameter (including pz of neutrino) only hadronic decays
histditaumassreginclnuhad = ROOT.TH1D("ditaumassreginclnuhad","di-#tau mass using regression all decays all parameters hadronic decays",100,0,100)
histditaumassreginclnuhad.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreginclnuhad.GetYaxis().SetTitle("number of occurence")
histditaumassreginclnuhad.GetYaxis().SetTitleOffset(1.5)
histditaumassreginclnuhad.SetLineColor(4)
histditaumassreginclnuhad.SetStats(0)
histditaumassreginclnuhadout = ROOT.TH1D("ditaumassreginclnuhadout","zzzz",100,0,100)
histditaumassreginclnuhadout.SetLineColor(2)
histditaumassreginclnuhadout.SetStats(0)



#histogram of ditau mass with regression all particles with 4Vector (all decay-products)
histditaumassregtotal = ROOT.TH1D("ditaumassregtotal","machine learning example using di-#tau 4Vector ",100,0,100)
histditaumassregtotal.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregtotal.GetYaxis().SetTitle("number of occurence")
histditaumassregtotal.GetYaxis().SetTitleOffset(1.5)
histditaumassregtotal.SetLineColor(4)
histditaumassregtotal.SetStats(0)
histditaumassregtotalout = ROOT.TH1D("ditaumassregtotalout","ssss",100,0,100)
histditaumassregtotalout.SetLineColor(2)
histditaumassregtotalout.SetStats(0)
histditaumassregtotalcalc = ROOT.TH1D("ditaumassregtotalcalc","ssss",100,0,100)
histditaumassregtotalcalc.SetLineColor(3)
histditaumassregtotalcalc.SetLineStyle(2)
histditaumassregtotalcalc.SetStats(0)

#histogram of ditau mass with regression all particles with 4Vector (all decay-products) only hadronic decays
histditaumassregtotalhad = ROOT.TH1D("ditaumassregtotalhad","di-#tau mass using regression all decays 4Vector all decay products hadronic decays",100,0,100)
histditaumassregtotalhad.GetXaxis().SetTitle("di-#tau mass")
histditaumassregtotalhad.GetYaxis().SetTitle("number of occurence")
histditaumassregtotalhad.SetLineColor(4)
histditaumassregtotalhad.SetStats(0)
histditaumassregtotalouthad = ROOT.TH1D("ditaumassregtotalouthad","ssss",100,0,100)
histditaumassregtotalouthad.SetLineColor(2)
histditaumassregtotalouthad.SetStats(0)
histditaumassregtotalcalchad = ROOT.TH1D("ditaumassregtotalcalchad","ssss",100,0,100)
histditaumassregtotalcalchad.SetLineColor(3)
histditaumassregtotalcalchad.SetLineStyle(2)
histditaumassregtotalcalchad.SetStats(0)


#histogram of ditau theta with regression all particles with 4Vector (all decay-products)
histditauregtheta = ROOT.TH1D("ditauregtheta","machine learning example using di-#tau 4Vector",100,0,3.5)
histditauregtheta.GetXaxis().SetTitle("di-#tau #theta")
histditauregtheta.GetYaxis().SetTitle("number of occurence")
histditauregtheta.GetYaxis().SetTitleOffset(1.2)
histditauregtheta.SetLineColor(4)
histditauregtheta.SetStats(0)
histditauregthetaout = ROOT.TH1D("ditauregthetaout","ssss",100,0,3.5)
histditauregthetaout.SetLineColor(2)
histditauregthetaout.SetStats(0)


#histogram of ditau vismass with regression all particles with 4Vector (all decay-products)
histditauvismassregall = ROOT.TH1D("ditauvismassregall","machine learning example using di-#tau 4Vector",100,0,100)
histditauvismassregall.GetXaxis().SetTitle("di-#tau vis-mass [GeV]")
histditauvismassregall.GetYaxis().SetTitle("number of occurence")
histditauvismassregall.GetYaxis().SetTitleOffset(1.5)
histditauvismassregall.SetLineColor(2)
histditauvismassregall.SetStats(0)
histditauvismassregallout = ROOT.TH1D("ditauvismassregallout","ssss",100,0,100)
histditauvismassregallout.SetLineColor(4)
histditauvismassregallout.SetStats(0)
histditauvismassregallcalc = ROOT.TH1D("ditauvismassregallcalc","tttt",100,0,100)
histditauvismassregallcalc.SetLineColor(3)
histditauvismassregallcalc.SetLineStyle(2)
histditauvismassregallcalc.SetStats(0)


#histogram of ditau vismass with regression only hadronic decays
histditauvismassreghad = ROOT.TH1D("ditauvismassreghad","di-#tau vis-mass using regression only hadronic decays",100,0,100)
histditauvismassreghad.GetXaxis().SetTitle("di-#tau vis-mass")
histditauvismassreghad.GetYaxis().SetTitle("number of occurence")
histditauvismassreghad.SetLineColor(2)
histditauvismassreghad.SetStats(0)
histditauvismassreghadout = ROOT.TH1D("ditauvismassreghadout","ssss",100,0,100)
histditauvismassreghadout.SetLineColor(4)
histditauvismassreghadout.SetStats(0)
histditauvismassreghadcalc = ROOT.TH1D("ditauvismassreghadcalc","tttt",100,0,100)
histditauvismassreghadcalc.SetLineColor(3)
histditauvismassreghadcalc.SetLineStyle(2)
histditauvismassreghadcalc.SetStats(0)


#histogram of px of ditau comparison train and test
histpxditautrain = ROOT.TH1D("pxditautrain","px di-#tau",100,-100,100)
histpxditautrain.GetXaxis().SetTitle("di-#tau px [GeV]")
histpxditautrain.GetYaxis().SetTitle("number of occurence")
histpxditautrain.SetLineColor(4)
histpxditautrain.SetStats(0)
histpxditautest = ROOT.TH1D("pxditautest","px sdsd",100,-100,100)
histpxditautest.SetLineColor(2)
histpxditautest.SetStats(0)

#histogram of py of ditau comparison train and test
histpyditautrain = ROOT.TH1D("pyditautrain","py di-#tau",100,-100,100)
histpyditautrain.GetXaxis().SetTitle("di-#tau py [GeV]")
histpyditautrain.GetYaxis().SetTitle("number of occurence")
histpyditautrain.SetLineColor(4)
histpyditautrain.SetStats(0)
histpyditautest = ROOT.TH1D("pyditautest","py sdsd",100,-100,100)
histpyditautest.SetLineColor(2)
histpyditautest.SetStats(0)

#histogram of pz of ditau comparison train and test
histpzditautrain = ROOT.TH1D("pzditautrain","pz di-#tau",100,-100,100)
histpzditautrain.GetXaxis().SetTitle("di-#tau pz[GeV]")
histpzditautrain.GetYaxis().SetTitle("number of occurence")
histpzditautrain.SetLineColor(4)
histpzditautrain.SetStats(0)
histpzditautest = ROOT.TH1D("pzditautest","pz sdsd",100,-100,100)
histpzditautest.SetLineColor(2)
histpzditautest.SetStats(0)

#histogram of E of ditau comparison train and test
histeditautrain = ROOT.TH1D("editautrain","energy di-#tau",100,0,150)
histeditautrain.GetXaxis().SetTitle("di-#tau E [GeV]")
histeditautrain.GetYaxis().SetTitle("number of occurence")
histeditautrain.SetLineColor(4)
histeditautrain.SetStats(0)
histeditautest = ROOT.TH1D("editautest","E sdsd",100,0,150)
histeditautest.SetLineColor(2)
histeditautest.SetStats(0)

#histogram of theta of ditau comparison train and test
histthetaditautrain = ROOT.TH1D("thetaditautrain","#theta di-#tau",100,0,3.5)
histthetaditautrain.GetXaxis().SetTitle("di-#tau #theta")
histthetaditautrain.GetYaxis().SetTitle("number of occurence")
histthetaditautrain.SetLineColor(4)
histthetaditautrain.SetStats(0)
histthetaditautest = ROOT.TH1D("thetaditautest","#theta sdsd",100,0,3.5)
histthetaditautest.SetLineColor(2)
histthetaditautest.SetStats(0)


#histogram of ditau playing with different parameters
histditauplay = ROOT.TH1D("ditauplay","machine learning example using di-#tau 4Vector squared entries ",200,0,10000)
histditauplay.GetXaxis().SetTitle("di-#tau m*m")
histditauplay.GetYaxis().SetTitle("number of occurence")
histditauplay.GetYaxis().SetTitleOffset(1.5)
histditauplay.SetLineColor(4)
histditauplay.SetStats(0)
histditauplayout = ROOT.TH1D("ditauplayout","ssss",200,0,10000)
histditauplayout.SetLineColor(2)
histditauplayout.SetLineStyle(2)
histditauplayout.SetStats(0)
histditauplaycalc = ROOT.TH1D("ditauplaycalc","ssss",200,0,10000)
histditauplaycalc.SetLineColor(3)
histditauplaycalc.SetLineStyle(2)
histditauplaycalc.SetStats(0)

"""
for i in train_px:
    histpxditautrain.Fill(i)
for j in train_py:
    histpyditautrain.Fill(j)
for k in train_pz:
    histpzditautrain.Fill(k)
for m in train_E:
    histeditautrain.Fill(m)
for s in train_theta:
    histthetaditautrain.Fill(s)
for n in test_px:
    histpxditautest.Fill(n)
for q in test_py:
    histpyditautest.Fill(q)
for c in test_pz:
    histpzditautest.Fill(c)
for v in test_E:
    histeditautest.Fill(v)
for w in test_theta:
    histthetaditautest.Fill(w)
"""

#n=len(train_output_all)


"""
# evaluate model
estimator = KerasRegressor(build_fn=ditaumass_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n,n_folds=10,random_state=seed)
results = cross_val_score(estimator, train_input_all, train_output_all, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=ditaumass_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n,n_folds=10, random_state=seed)
results = cross_val_score(pipeline, train_input_all, train_output_all, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""

def mass_model_had(batch_size,epochs):
    mass_model_had = Sequential()
    mass_model_had.add(Dense(40,input_dim=7,kernel_initializer='random_uniform',activation='relu'))
    mass_model_had.add(Dense(20,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.add(Dense(10,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.add(Dense(1,kernel_initializer='random_uniform',activation = 'relu'))
    mass_model_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_had.fit(train_input,train_output,batch_size,epochs,verbose=1)
    mass_score_had = mass_model_had.evaluate(test_input,test_output,batch_size,verbose=0)
    ditaumass_had = mass_model_had.predict(test_input,batch_size,verbose=0)
    print "mass_model_had(",batch_size,epochs,")"
    print "score for hadronic decays:",mass_score_had
    #preparing the histograms
    for i in ditaumass_had:
        histditaumassreg.Fill(i)
    for k in test_output:
        histditaumassregout.Fill(k)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression hadronic")
    max_bin = max(histditaumassreg.GetMaximum(),histditaumassregout.GetMaximum())
    histditaumassreg.SetMaximum(max_bin*1.08)
    histditaumassreg.Draw()
    histditaumassregout.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassreg,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregout,"actual mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_hadronic.png")

def mass_model_all(batch_size,epochs):
    mass_model_all = Sequential()
    mass_model_all.add(Dense(40,input_dim=6,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    mass_model_all.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model_all.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_all.fit(train_input_all,train_output_all,batch_size,epochs,verbose=1)
    mass_score_all = mass_model_all.evaluate(mass_input_all,mass_output_all,batch_size,verbose=0)
    ditaumass_all = mass_model_all.predict(test_input_all,batch_size,verbose=0)
    ditaumass_all_overfit = mass_model_all.predict(train_input_all[0:len(test_output_all),:],batch_size,verbose=0)
    print "mass_model_all(",batch_size,epochs,")"
    print "score for all decays:",mass_score_all
    #preparing the histograms
    for j in ditaumass_all:
        histditaumassregall.Fill(j)
    for h in test_output_all:
        histditaumassregallout.Fill(h)
    for d in ditaumass_all_overfit:
        histditaumassregalloverfit.Fill(d)
    #histogram of di-tau mass using regression all decays
    canv = ROOT.TCanvas("di-tau mass using regression all decays")
    max_bin = max(histditaumassregall.GetMaximum(),histditaumassregallout.GetMaximum(),histditaumassregalloverfit.GetMaximum())
    histditaumassregall.SetMaximum(max_bin*1.08)
    histditaumassregall.Draw()
    histditaumassregallout.Draw("SAME")
    #histditaumassregalloverfit.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregall,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregallout,"actual mass","PL")
    #leg.AddEntry(histditaumassregalloverfit,"overfit-test","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_all.png")   

def mass_model_inclnu(batch_size,epochs):
    mass_model_inclnu = Sequential()
    mass_model_inclnu.add(Dense(40,input_dim=8,kernel_initializer='random_uniform',activation='relu'))
    mass_model_inclnu.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    mass_model_inclnu.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    mass_model_inclnu.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    mass_model_inclnu.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    mass_model_inclnu.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_inclnu.fit(train_input_inclnu,train_output_inclnu,batch_size,epochs,verbose=1)
    mass_score_inclnu = mass_model_inclnu.evaluate(test_input_inclnu,test_output_inclnu,batch_size,verbose=0)
    ditaumass_inclnu = mass_model_inclnu.predict(test_input_inclnu,batch_size,verbose=0)
    print "mass_model_inclnu(",batch_size,epochs,")"
    print "score:",mass_score_inclnu
    #preparing the histograms
    for k in ditaumass_inclnu:
        histditaumassreginclnu.Fill(k)
    for j in test_output_inclnu:
        histditaumassreginclnuout.Fill(j)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays all parameters")
    max_bin = max(histditaumassreginclnu.GetMaximum(),histditaumassreginclnuout.GetMaximum())
    histditaumassreginclnu.SetMaximum(max_bin*1.08)
    histditaumassreginclnu.Draw()
    histditaumassreginclnuout.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassreginclnu,"reconstructed mass","PL")
    leg.AddEntry(histditaumassreginclnuout,"actual mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_inclnu.png")

def mass_model_inclnu_had(batch_size,epochs):
    mass_model_inclnu_had = Sequential()
    mass_model_inclnu_had.add(Dense(40,input_dim=8,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(20,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(5,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.add(Dense(1,kernel_initializer='normal',activation='relu'))
    mass_model_inclnu_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_inclnu_had.fit(train_input_inclnu,train_output_inclnu,batch_size,epochs,verbose=1)
    mass_score_inclnu_had = mass_model_inclnu_had.evaluate(test_input_inclnu,test_output_inclnu,batch_size,verbose=0)
    ditaumass_inclnu = mass_model_inclnu_had.predict(test_input_inclnu,batch_size,verbose=0)
    print "mass_model_inclnu_had(",batch_size,epochs,")"
    print "score:",mass_score_inclnu_had
    #preparing the histograms
    for k in ditaumass_inclnu_had:
        histditaumassreginclnuhad.Fill(k)
    for p in test_output_inclnu:
        histditaumassreginclnuhadout.Fill(p)
    #histogram of di-tau mass using regression all decays all parameters
    canv = ROOT.TCanvas("di-tau mass using regression all decays all parameters hadronic decays")
    max_bin = max(histditaumassreginclnuhad.GetMaximum(),histditaumassreginclnuhadout.GetMaximum())
    histditaumassreginclnuhad.SetMaximum(max_bin*1.08)
    histditaumassreginclnuhad.Draw()
    histditaumassreginclnuhadout.Draw("SAME")
    leg = ROOT.TImage.Create()
    leg.AddEntry(histditaumassreginclnuhad,"reconstructed mass","PL")
    leg.AddEntry(histditaumassreginclnuhadout,"actual mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_inclnu_had.png")


def mass_model_total(batch_size,epochs):
    mass_model_total = Sequential()
    mass_model_total.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    mass_model_total.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    mass_model_total.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    mass_model_total.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    mass_model_total.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    #mass_model_total.load_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/mass_model_total_weights_try2.h5")
    mass_model_total.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total.fit(train_input_total,train_output_total,batch_size,epochs,verbose=1)
    mass_score_total = mass_model_total.evaluate(test_input_total,test_output_total,batch_size,verbose=0)
    ditaumass_total = mass_model_total.predict(test_input_total,batch_size,verbose=0)
    #print "try4"
    mass_model_total.summary()
    print "mass_model_total(",batch_size,epochs,")"
    print "score:",mass_score_total
    mass_model_total.save_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/mass_model_total_weights.h5")
    #preparing the histograms
    for u in ditaumass_total:
        histditaumassregtotal.Fill(u)
    for g in test_output_total:
        histditaumassregtotalout.Fill(g)
    for i in range(0,test_length_total):
        mass_calc = ROOT.TLorentzVector(test_input_total[i,0],test_input_total[i,1],test_input_total[i,2],test_input_total[i,3]).M()
        histditaumassregtotalcalc.Fill(mass_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products")
    max_bin = max(histditaumassregtotal.GetMaximum(),histditaumassregtotalout.GetMaximum(),histditaumassregtotalcalc.GetMaximum())
    histditaumassregtotal.SetMaximum(max_bin*1.08)
    histditaumassregtotal.Draw()
    histditaumassregtotalout.Draw("SAME")
    histditaumassregtotalcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregtotal,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregtotalout,"actual mass","PL")
    leg.AddEntry(histditaumassregtotalcalc,"calculated mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_total.png")

def mass_model_total_had(batch_size,epochs):
    mass_model_total_had = Sequential()
    mass_model_total_had.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    #mass_model_total_had.add(BatchNormalization())
    mass_model_total_had.compile(loss='mean_squared_error',optimizer='adam')
    mass_model_total_had.fit(train_input_total,train_output_total,batch_size,epochs,verbose=1)
    mass_score_total_had = mass_model_total_had.evaluate(test_input_total,test_output_total,verbose=0)
    ditaumass_total_had = mass_model_total_had.predict(test_input_total,batch_size,verbose=0)
    print "mass_model_total_had(",batch_size,epochs,")"
    print "score:",mass_score_total_had
    #preparing the histograms
    for u in ditaumass_total_had:
        histditaumassregtotalhad.Fill(u)
    for g in test_output_total_had:
        histditaumassregtotalout_had.Fill(g)
    for i in range(0,len(test_output_total_had)):
        mass_calc = ROOT.TLorentzVector(test_input_total[i,0],test_input_total[i,1],test_input_total[i,2],test_input_total[i,3]).M()
        histditaumassregtotalcalc.Fill(mass_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products hadronic decays")
    max_bin = max(histditaumassregtotalhad.GetMaximum(),histditaumassregtotalouthad.GetMaximum(),histditaumassregtotalcalchad.GetMaximum())
    histditaumassregtotalhad.SetMaximum(max_bin*1.08)
    histditaumassregtotalhad.Draw()
    histditaumassregtotalouthad.Draw("SAME")
    histditaumassregtotalcalchad.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditaumassregtotalhad,"reconstructed mass","PL")
    leg.AddEntry(histditaumassregtotalouthad,"actual mass","PL")
    leg.AddEntry(histditaumassregtotalcalchad,"calculated mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_total.png")


def theta_model(batch_size,epochs):
    theta_model = Sequential()
    theta_model.add(Dense(20,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    theta_model.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    theta_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    theta_model.compile(loss='mean_squared_error',optimizer='adam')
    theta_model.fit(train_input_theta,train_output_theta,batch_size,epochs,verbose=1)
    theta_score = theta_model.evaluate(test_input_theta,test_output_theta,batch_size,verbose=0)
    ditautheta = theta_model.predict(test_input_theta,batch_size,verbose=0)
    print "theta_model(",batch_size,epochs,")"
    print "score:",theta_score
    #preparing the histograms
    for t in ditautheta:
        histditauregtheta.Fill(t)
    for d in test_output_theta:
        histditauregthetaout.Fill(d)
    #histograms
    canv = ROOT.TCanvas("di-tau theta using regression all decays 4Vector all decay products")
    max_bin = max(histditauregtheta.GetMaximum(),histditauregthetaout.GetMaximum())
    histditauregtheta.SetMaximum(max_bin*1.3)
    histditauregtheta.Draw()
    histditauregthetaout.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.8,0.9,0.9)
    leg.AddEntry(histditauregtheta,"reconstructed theta","PL")
    leg.AddEntry(histditauregthetaout,"actual theta","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_theta.png")

def vismass_model_all(batch_size,epochs):
    vismass_model_all = Sequential()
    vismass_model_all.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(5,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_all.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    vismass_model_all.compile(loss='mean_squared_error',optimizer='adam')
    vismass_model_all.fit(train_input_vismass_all,train_output_vismass_all,batch_size,epochs,verbose=1)
    vismass_score_all = vismass_model_all.evaluate(test_input_vismass_all,test_output_vismass_all,batch_size,verbose=0)
    ditauvismass_all = vismass_model_all.predict(test_input_vismass_all,batch_size,verbose=0)
    print "vismass_model_all(",batch_size,epochs,")"
    print "score:",vismass_score_all
    #preparing the histograms
    for p in ditauvismass_all:
        histditauvismassregall.Fill(p)
    for s in test_output_vismass_all:
        histditauvismassregallout.Fill(s)
    for i in range(0,len(dataset_vismass_test_all)):
        vismass_calc = ROOT.TLorentzVector(dataset_vismass_test_all[i,0],dataset_vismass_test_all[i,1],dataset_vismass_test_all[i,2],dataset_vismass_test_all[i,3]).M()
        histditauvismassregallcalc.Fill(vismass_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau vismass regression all")
    max_bin = max(histditauvismassregall.GetMaximum(),histditauvismassregallout.GetMaximum(),histditauvismassregallcalc.GetMaximum())
    histditauvismassregall.SetMaximum(max_bin*1.2)
    histditauvismassregall.Draw()
    histditauvismassregallout.Draw("SAME")
    histditauvismassregallcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.8,0.9,0.9)
    leg.AddEntry(histditauvismassregall,"reconstructed vis-mass","PL")
    leg.AddEntry(histditauvismassregallout,"actual vis-mass","PL")
    leg.AddEntry(histditauvismassregallcalc,"calculated vis-mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("vismass_ditau_all.png")

def vismass_model_had(batch_size,epochs):
    vismass_model_had = Sequential()
    vismass_model_had.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(5,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.add(Dense(1,kernel_initializer='random_uniform',activation='relu'))
    vismass_model_had.compile(loss='mean_squared_error',optimizer='adam')
    vismass_model_had.fit(train_input_vismass_had,train_output_vismass_had,batch_size,epochs,verbose=1)
    vismass_score_had = vismass_model_had.evaluate(test_input_vismass_had,test_output_vismass_had,batch_size,verbose=0)
    ditauvismass_had = vismass_model_had.predict(test_input_vismass_had,batch_size,verbose=0)
    print "vismass_model_had(",batch_size,epochs,")"
    print "score:",vismass_score_had
    for a in ditauvismass_had:
        histditauvismassreghad.Fill(a)
    for w in test_output_vismass_had:
        histditauvismassreghadout.Fill(w)
    for i in range(0,len(dataset_vismass_test_had)):
        vismass_calc_had = ROOT.TLorentzVector(dataset_vismass_test_had[i,0],dataset_vismass_test_had[i,1],dataset_vismass_test_had[i,2],dataset_vismass_test_had[i,3]).M()
        histditauvismassreghadcalc.Fill(vismass_calc_had)
    #histogram of di-tau vismass using regression only hadronic decays
    canv = ROOT.TCanvas("di-tau vismass regression had")
    max_bin = max(histditauvismassreghad.GetMaximum(),histditauvismassreghadout.GetMaximum(),histditauvismassreghadcalc.GetMaximum())
    histditauvismassreghad.SetMaximum(max_bin*1.2)
    histditauvismassreghad.Draw()
    histditauvismassreghadout.Draw("SAME")
    histditauvismassreghadcalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.8,0.9,0.9)
    leg.AddEntry(histditauvismassreghad,"reconstructed vis-mass","PL")
    leg.AddEntry(histditauvismassreghadout,"actual vis-mass","PL")
    leg.AddEntry(histditauvismassreghadcalc,"calculated vis-mass","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("vismass_ditau_had.png")

def play_model(batch_size,epochs):
    #clf_play = RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=0)
    #clf_play.fit(train_input_play,train_output_play)
    #scores_play = clf_play.score(test_input_play,test_output_play)
    #ditau_play = clf_play.predict(test_input_play)
    #print "score RandomForest total:",scores_play
    play_model = Sequential()
    play_model.add(Dense(40,input_dim=4,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(50,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(40,kernel_initializer='random_uniform',activation='relu'))
    #play_model.add(Dense(30,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(20,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(10,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(5,kernel_initializer='random_uniform',activation='relu'))
    play_model.add(Dense(1,kernel_initializer='random_uniform',activation='linear'))
    #play_model.load_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/play_model_weights.h5")
    play_model.compile(loss='mean_squared_error',optimizer='adam')
    play_model.fit(train_input_play,train_output_play,batch_size,epochs,verbose=1)
    play_score = play_model.evaluate(test_input_play,test_output_play,batch_size,verbose=0)
    ditau_play = play_model.predict(test_input_play,batch_size,verbose=0)
    play_model.summary()
    play_model.save_weights("/mnt/t3nfs01/data01/shome/pbaertsc/tauregression/CMSSW_8_0_23/src/play_model_weights.h5")
    print "play_model(",batch_size,epochs,")"
    print "score:",play_score
    #preparing the histograms
    for u in ditau_play:
        histditauplay.Fill(u)
    for g in test_output_play:
        histditauplayout.Fill(g)
    for i in range(0,len(test_output_play)):
        play_calc = test_input_play[i,3]-test_input_play[i,0]-test_input_play[i,1]-test_input_play[i,2]
        histditauplaycalc.Fill(play_calc)
    #histograms
    canv = ROOT.TCanvas("di-tau mass using regression all decays 4Vector all decay products")
    max_bin = max(histditauplay.GetMaximum(),histditauplayout.GetMaximum(),histditauplaycalc.GetMaximum())
    histditauplay.SetMaximum(max_bin*1.08)
    histditauplay.Draw()
    histditauplayout.Draw("SAME")
    histditauplaycalc.Draw("SAME")
    leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
    leg.AddEntry(histditauplay,"reconstructed m*m","PL")
    leg.AddEntry(histditauplayout,"actual m*m","PL")
    leg.AddEntry(histditauplaycalc,"calculated m*m","PL")
    leg.Draw()
    img = ROOT.TImage.Create()
    img.FromPad(canv)
    img.WriteImage("reg_ditau_mass_playing_try9.png")

####put wanted model here ####
mass_model_total(15,20)
#mass_model_had(10,20)
#mass_model_inclnu(15,30)
#theta_model(20,20)
#vismass_model_all(20,20)
#mass_model_all(15,20)
#play_model(20,20)

"""
#histogram of di-tau px train and test
canv6 = ROOT.TCanvas("di-tau px comparison train and test")
max_bin6 = max(histpxditautrain.GetMaximum(),histpxditautest.GetMaximum())
histpxditautrain.SetMaximum(max_bin6*1.08)
histpxditautrain.Draw()
histpxditautest.Draw("SAME")

leg6 = ROOT.TLegend(0.6,0.7,0.9,0.9)
leg6.AddEntry(histpxditautrain,"px train sample","PL")
leg6.AddEntry(histpxditautest,"px test sample","PL")
leg6.Draw()

canv6.Write()
img6 = ROOT.TImage.Create()
img6.FromPad(canv6)
img6.WriteImage("px_ditau_comp.png")

#histogram of di-tau py train and test
canv7 = ROOT.TCanvas("di-tau pxy comparison train and test")
max_bin7 = max(histpyditautrain.GetMaximum(),histpyditautest.GetMaximum())
histpyditautrain.SetMaximum(max_bin7*1.08)
histpyditautrain.Draw()
histpyditautest.Draw("SAME")

leg7 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg7.AddEntry(histpyditautrain,"py train sample","PL")
leg7.AddEntry(histpyditautest,"py test sample","PL")
leg7.Draw()

canv7.Write()
img7 = ROOT.TImage.Create()
img7.FromPad(canv7)
img7.WriteImage("py_ditau_comp.png")

#histogram of di-tau pz train and test
canv8 = ROOT.TCanvas("di-tau pz comparison train and test")
max_bin8 = max(histpzditautrain.GetMaximum(),histpzditautest.GetMaximum())
histpzditautrain.SetMaximum(max_bin8*1.08)
histpzditautrain.Draw()
histpzditautest.Draw("SAME")

leg8 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg8.AddEntry(histpzditautrain,"pz train sample","PL")
leg8.AddEntry(histpzditautest,"pz test sample","PL")
leg8.Draw()

canv8.Write()
img8 = ROOT.TImage.Create()
img8.FromPad(canv8)
img8.WriteImage("pz_ditau_comp.png")

#histogram of di-tau energy train and test
canv9 = ROOT.TCanvas("di-tau energy comparison train and test")
max_bin9 = max(histeditautrain.GetMaximum(),histeditautest.GetMaximum())
histeditautrain.SetMaximum(max_bin9*1.08)
histeditautrain.Draw()
histeditautest.Draw("SAME")

leg9 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg9.AddEntry(histeditautrain,"energy train sample","PL")
leg9.AddEntry(histeditautest,"energy test sample","PL")
leg9.Draw()

canv9.Write()
img9 = ROOT.TImage.Create()
img9.FromPad(canv9)
img9.WriteImage("e_ditau_comp.png")

#histogram of di-tau theta train and test
canv10 = ROOT.TCanvas("di-tau theta comparison train and test")
max_bin10 = max(histthetaditautrain.GetMaximum(),histthetaditautest.GetMaximum())
histthetaditautrain.SetMaximum(max_bin10*1.2)
histthetaditautrain.Draw()
histthetaditautest.Draw("SAME")

leg10 = ROOT.TLegend(0.6,0.8,0.9,0.9)
leg10.AddEntry(histthetaditautrain,"theta train sample","PL")
leg10.AddEntry(histthetaditautest,"theta test sample","PL")
leg10.Draw()

canv10.Write()
img10 = ROOT.TImage.Create()
img10.FromPad(canv10)
img10.WriteImage("theta_ditau_comp.png")
"""

