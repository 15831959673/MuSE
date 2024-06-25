# MuSE
MuSE:A Deep Learning Model based on Multi-Feature Fusion for Super-Enhancer Prediction


## 1. File descriptions

### human_model/mouse_model

#### data_preparation.py

Data processing for loading X2 features.

#### run.py

Load the model and the remaining two features, and calculate the evaluation metrics.

### model

The trained human and mouse model parameters are saved in this file.

## 2. Environment setup

We recommend you to build a python virtual environment with Anaconda. 


#### 2.1 Create and activate a new virtual environment

```
conda create -n MuSE python=3.8
conda activate MuSE
```



#### 2.2 Install the package and other requirements

```
python -m pip install -r requirements.txt
```
If you want to download the pytorch environment separately, execute this command
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```



## 3. Calculate evaluation indicators
### human/mouse
- *python run.py* 
