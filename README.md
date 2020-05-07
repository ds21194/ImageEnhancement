# ImageEnhancement

Image Enhancement using Machine Learning Models of Neural Networks

I used the neural network architecture suggested in [this](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) paper, for Image Enhancement instead of Image recognition. 

The result are not as good as I expected. 

# How to use?

There are two functionalities: Image deblur and Image denoise.

You can train the model you choose with the functions under "ModelTrainer". It will create your model from scratch.

You can also choose an already trained model. 
(I will try to improve those models with time to bring better results)

Inside "Image Enhancement" there is a function to load a trained model.
Use it to load one of the models under "models" (models which I trained with my gpu)
or models under "savedModels" (models I trained with google colab)
Than you can use the function "restore_image" or "restore_image_v2"
(they function in different ways)

*for now this works only for grayscale images.

###### You welcome to send me privatly or upload new barch with approvements, or even only to suggest a better way if you have any idea :) 
