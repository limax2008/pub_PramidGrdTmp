
# pub_PramidGrdTmp

## Introduction
This is one implementation for our paper "Anomaly Detection with Multi-scale Pyramid Grid Templates".

## How to run the code
The main function is tst_one.py.  
The method is based on optical flow, therefore you should computed it first and store it in one folder.
And then set the optical flow stored path to the item "optTstPth" in the config.ini.

The provided demo is based on UCSD ped1 dataset, if you want to run the code for your dataset, you should first train the multi-scale maximum pyramid template first, store the template in a suitable path, and set the path to item "tmpPth" in config.ini.

Currently, you can run the code tst_one.py directly using the uploaded test frames. If you have any problem, please contact us by email "lishifeng2007@gmail.com".



