# tau-regression


access to PSI CmsTier3 is required to get the inputs for the neural network and SVfit

run nn_github.py in virtual env with pandas, tensorflow and keras installed. (GPU is recommended)

run nnplots_github.py in CMSSW_8_0_23

Running SVfit is only recommended if there is access to 10 or more CPU cores.
Set the library bath: export LD_LIBRARY_PATH=${PWD}/../svFit/lib:${LD_LIBRARY_PATH} and 
run svfit_github.py in CMSSW_8_0_23
