import numpy
import pandas
import ROOT
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
#from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#load dataset
dataframe_mass_train = pandas.read_csv("train_reg_ditau_mass.csv",delim_whitespace=False,header=None)
dataset_mass_train = dataframe_mass_train.values
dataframe_mass_test= pandas.read_csv("test_reg_ditau_mass.csv",delim_whitespace=False,header=None)
dataset_mass_test = dataframe_mass_test.values
dataframe_mass= pandas.read_csv("reg_ditau_mass.csv",delim_whitespace=False,header=None)
dataset_mass = dataframe_mass.values

dataframe_mass_train_all = pandas.read_csv("train_reg_ditau_mass_all.csv",delim_whitespace=False,header=None)
dataset_mass_train_all = dataframe_mass_train_all.values
dataframe_mass_test_all= pandas.read_csv("test_reg_ditau_mass_all.csv",delim_whitespace=False,header=None)
dataset_mass_test_all = dataframe_mass_test_all.values
dataframe_mass_all= pandas.read_csv("reg_ditau_mass_all.csv",delim_whitespace=False,header=None)
dataset_mass_all = dataframe_mass_all.values



# split into input and output variables
train_input = dataset_mass_train[:,0:5]
train_output = dataset_mass_train[:,5]
test_input = dataset_mass_test[:,0:5]
test_output = dataset_mass_test[:,5]
mass_input = dataset_mass[:,:]

train_input_all = dataset_mass_train_all[:,0:5]
train_output_all = dataset_mass_train_all[:,5]
test_input_all = dataset_mass_test_all[:,0:5]
test_output_all = dataset_mass_test_all[:,5]
mass_input_all = dataset_mass_all[:,:]

#histogram of ditau mass with regression only hadronic decays
histditaumassreg = ROOT.TH1D("ditaumassreg","di-#tau mass using regression hadronic decays",100,0,100)
histditaumassreg.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassreg.GetYaxis().SetTitle("number of occurence")

#histogram of ditau mass with regression all particles
histditaumassregall = ROOT.TH1D("ditaumassregall","di-#tau mass using regression all decays",100,0,100)
histditaumassregall.GetXaxis().SetTitle("di-#tau mass [GeV]")
histditaumassregall.GetYaxis().SetTitle("number of occurence")


mass_model = Sequential()
mass_model.add(Dense(5,input_dim=5,kernel_initializer='normal',activation='relu'))
mass_model.add(Dense(1,kernel_initializer='normal'))
mass_model.compile(loss='mean_squared_error',optimizer='adam')
mass_model.fit(train_input,train_output,batch_size=5,epochs=100,verbose=0)
mass_score = mass_model.evaluate(test_input,test_output,verbose=0)
ditaumass = mass_model.predict(mass_input,batch_size=5,verbose=0)

mass_model_all = Sequential()
mass_model_all.add(Dense(5,input_dim=5,kernel_initializer='normal',activation='relu'))
mass_model_all.add(Dense(1,kernel_initializer='normal'))
mass_model_all.compile(loss='mean_squared_error',optimizer='adam')
mass_model_all.fit(train_input_all,train_output_all,batch_size=5,epochs=100,verbose=0)
mass_score_all = mass_model_all.evaluate(test_input_all,test_output_all,verbose=0)
ditaumass_all = mass_model_all.predict(mass_input_all,batch_size=5,verbose=0)

for i in ditaumass:
    histditaumassreg.Fill(i)

for j in ditaumass_all:
    histditaumassregall.Fill(j)


#histogram of di-tau mass using regression hadronic decays
canv1 = ROOT.TCanvas("di-tau mass using regression hadronic")
histditaumassreg.Draw()

#histogram of di-tau mass using regression all decays
canv2 = ROOT.TCanvas("di-tau mass using regression all decays")
histditaumassregall.Draw()

canv2.WaitPrimitive()
