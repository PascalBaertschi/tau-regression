# tau-regression


Access to PSI CmsTier3 is required to get the inputs for the neural network and SVfit

## Running training and evaluation

Run [nn_github.py](nn_github.py) in virtualenv with pandas, tensorflow and keras installed (GPU is recommended):

If not yet done, install miniconda following the instructions, no changes to default arguments:

```shell
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh
```

Then create the virtualenv with the required packages:

```shell
conda create --name=taureg python=2 keras tensorflow pandas scikit-learn
source activate taureg
```

Run `python nn_github.py`.

## Plots

Run [nnplots_github.py](nnplots_github.py) in `CMSSW_8_0_23`:

Setup:

```shell
cmsrel CMSSW_8_0_23
cd CMSSW_8_0_23/src
cmsenv
```

## Running SVfit

Running SVfit is only recommended if there is access to 10 or more CPU cores.

Set the library path: export LD_LIBRARY_PATH=${PWD}/../svFit/lib:${LD_LIBRARY_PATH}` and
run [svfit_github.py](svfit_github.py) in `CMSSW_8_0_23` (see above).
