<< Objective >>
Train a Neural Network model to extract image embeddings from scratch. This model could be repeatedly used to provide meaningful and digitized image information. The produced embedding could be used as a input to other system.

Here we use Kaggle H&M data as dataset:
	
  H&M Personalized Fashion Recommendations 
  https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations



<< Codes structure >>

@ yuli_CNN_AE_base_resnet_main.py:
- Train a model with early stopping and save it.
- This contains functions to generate model performance examples.

@ yuli_CNN_AE_base_resnet_train.py
- Codes of train neural network with early stop.

@ yuli_CNN_AE_base_resnet_evaluation.py
- Codes of model evaluation class "EvalModel".

@ yuli_CNN_AE_base_resnet_NNmodel.py
- Codes of Auto Encoder model that based on ResNet structure, and the defined Dataset class.

@ yuli_CNN_AE_base_resnet_data.py
- Functions for data processing.

@ yuli_CNN_AE_base_resnet_config.py
- The configuration file. (Not open to public)
