# Image-Classification
This repository shows the steps required for Image Classification using Pretrained Models like InceptionResnet, Mobilenet etc on popular datasets like CIFAR10 and EMNIST. 


## Training Data
Run the train.py file to train. 
You can specify the transfer learning model and the dataset. 
Example command:  
python train.py -d EMNIST -m EfficientNetB7  
Parameters:  
-bsize: Specify the batch size  
-d: Dataset to use. (Specify folder name from dataset dir.)  
-m: Transfer Learning model to use  
--visualise: Flag to specify if you want to visualise some input data  

## Testing Data
Run the test.py file to test on sample images. 
You can specify the transfer learning model and the dataset. 
Example command:  
python test.py -p test_folder -d CIFAR10
Parameters:  
-p: Path to test dir consisting test images  
-d: Dataset to use. (Specify folder name from dataset dir.)  
-m: Transfer Learning model to use  
