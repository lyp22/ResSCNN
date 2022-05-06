# ResSCNN

A no-reference point cloud quality assessment method using the sparse convolutional neural network.

Enviroment
-----------

python 3.7  
cudatoolkit 11.1  
pytorch 1.8  
minkowskiengine 0.5  
xlrd 1.2  
open3d  
easydict  
numpy  
scipy  
tensorboardX  
pandas  
tqdm  

Sample scale normalization
-----------
The xyz coordinates of training samples are recommended to be casted into 0-2000. If not, voxel_size in config.py should be adjusted properly.

Usage
-----------
train.py is the main program. The explanations of some parameters in config.py are as below:

trainer_arg.add_argument('--train_file', type=str, default='./config/train.xlsx', help='file name and MOS for training set')  
trainer_arg.add_argument('--test_file', type=str, default='./config/test.xlsx', help='file name and MOS for testing set')  
trainer_arg.add_argument('--train_path', type=str, default='./config/path.xlsx', help='file name and file path for training set')  
trainer_arg.add_argument('--test_path', type=str, default='./config/path.xlsx', help='file name and file path for testing set')  



Bibtex
-----------
If you use this code please cite the paper

"Yipeng Liu, Yang Qi, Yiling Xu, Le Yang, "Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Approach", arXiv:2012.11895, 2021."

@article{liu2021LSPCQA,  
    title={Point Cloud Quality Assessment: Dataset Construction and Learning-based No-Reference Approach},  
    author={Yipeng Liu and Yang Qi and Yiling Xu and Le Yang},  
    journal={arXiv preprint arXiv:2012.11895},  
    year={2021}  
}

