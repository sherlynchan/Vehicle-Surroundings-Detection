# DS-GA-1008-deep-learning

This repo is for the final project of NYU DS-1008 Deep Learning.\
Project member: Shaoling Chen (sc6995), Shizhan Gong (sg5722), Yunan Hu (yh1844)

The object of the project is to detect vehicle's surroundings. 

We address this problem of recovering the top-down view of surroundings from six color images from cameras attached to the same car by proposing models to recover both the static objects (roadmap layout) and dynamic objects (traffic participants). 

Our approaches are as follows:
- We propose a [variational autoencoder networks](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/vae_roadmap) to predict the road map layout.
- We build and compare three different architectures,separately on top of [Yolo-V1](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/rotated_yolo_detector), [MonoLayout Model](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/monolayout_detector) and [Unet](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/unet_detector), to detect the surrounding objects.
- We further improve our models’ performance by solving [Jigsaw puzzles](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/jigsaw_pretraining) as a pre-training task.

Run [here](https://github.com/670973787/DS-GA-1008-deep-learning/tree/master/44-Onlooker-Round%202) to check our final scores.