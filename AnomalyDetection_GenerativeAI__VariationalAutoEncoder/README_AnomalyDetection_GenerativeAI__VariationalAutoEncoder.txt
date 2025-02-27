Here present both Variational AutoEncoder (VAE) and AutoEncoder (AE) in anomaly detection. For anomaly detection or fraud detection, VAE usually performs better than AE due to its probabilistic nature. 


A brief review of possible applications of each model:
- Variational AutoEncoder:
 Anomaly Detection, Generative AI (Generate Synthetic images or data), De-noisng.
- AutoEncoder:
 Anomaly Detection, Dimension Reduction (Generate representative embeddings), Image compression, De-noising.

---------------- Variational AutoEncoder ---------------
Objective: Train an Variational AutoEncoder NN model from scratch.

Codes structure:
@ yuli_CNN_AE_base_resnet_config.py
configuration file. (Not included here)

@ yuli_CNN_VAE_base_resnet_main.py:
Train a model & generate model performance examples.

@ yuli_CNN_VAE_base_resnet_train.py
Codes of train neural network with early stop.

@ yuli_CNN_VAE_base_resnet_evaluation.py
Codes of evaluation class "EvalModel"

@ yuli_CNN_VAE_base_resnet_NNmodel.py
Codes of Variational Auto Encoder model that based on ResNet,

@ yuli_CNN_VAE_base_resnet_NNdataset.py
Defined Dataset class for the VAE.

@ yuli_CNN_AE_base_resnet_data.py
Some auxiliary functions related to data processing.

---------------- AutoEncoder (AE) ---------------
Objective: Train an AutoEncoder NN model from scratch. 

Codes structure:
@ yuli_CNN_AE_base_resnet_config.py
configuration file. (Not included here)

@ yuli_CNN_AE_base_resnet_main.py:
Train a model & generate model performance examples.

@ yuli_CNN_AE_base_resnet_train.py
Codes of train neural network with early stop.

@ yuli_CNN_AE_base_resnet_evaluation.py
Codes of evaluation class "EvalModel"

@ yuli_CNN_AE_base_resnet_NNmodel.py
Codes of Auto Encoder model that based on ResNet,

@ yuli_CNN_AE_base_resnet_NNdataset.py
Defined Dataset class for the AE.

@ yuli_CNN_AE_base_resnet_data.py
- Some auxiliary functions related to data processing.

