# eMotion-GAN
This repository contains the source code for our paper:

**eMotion-GAN: Motion-based GAN for Photorealistic and Facial Expression Preserving Frontal View Synthesis**

![hippo](images/our_new_approach.png)
![hippo](images/anim.gif)

## **Updates**
- The full version of the code will be available soon!

## **Installation**
Create and activate conda environment:
```
conda create -n eMotionGAN python=3.10
conda activate eMotionGAN
```

Install all dependencies:
```
pip install -r requirements.txt
```

Install Jupyter Lab to visualize demo:
```
conda install -c conda-forge jupyterlab
```

### Demos
![hippo](anim.gif)

### Frontal View Synthesis (FVS) & Facial Expression Preserving 
![hippo](images/resu_compar.png)

### Cross-subject Facial Motion Transfer
![hippo](images/motion_transfer.png)

![hippo](images/anim_MT.gif)


## Training

By default, the training datasets are structured as follows:
```
├── datasets

    ├── CK+
        ├── sub_id
            ├── seq_id
              ├── img_001.png
              ├── img_002.png
              ...
              ├── img_010.png
              
    ├── ADFES
        ├── sub_id
            ├── seq_id
              ├── img_001.png
              ├── img_002.png
              ...
              ├── img_010.png
              
    ...
```
    
### Motion Calculation

```
python generate_data_files.py --dataset_name SNAP
                              --data_dir_F ./datasets/SNAPcam1/ 
                              --data_dir_NF ./datasets/SNAPcam2 
                              --save_dir_F ./datasets optical_flows/ 
                              --save_dir_NF ./datasets/optical_flows/ 
                              --emotions_file_path ./datasets/emotions/SNAP_emotions.csv 
                              --flow_algo Farneback
```

### training

Download [DMUE](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE) and its pre-trained model.

```
python train.py --data_path ./datasets/data_file.txt 
                --proto_path ./datasets/protocols/SNAP_proto.csv 
                --fold 1
```

## Evaluation

### Frontal View Synthesis
```
python FVS_demo.py --img1 images/images_1.png 
                   --img2 images/images2.png
                   --model_path weights/eMotionGAN_model.pth
```

### Motion Transfer
```
python motion_transfer_demo.py --img1 images/images_1.png 
                               --img2 images/images2.png 
                               --neutrl_img images/neutral_img.png
                               --model_path weights/eMotionGAN_model.pth
```


## Citation
If you find this repo useful, please consider citing our paper

```ref```

