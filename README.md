
# pub_PramidGrdTmp

## Introduction
This is one implementation for our paper "Anomaly Detection with Multi-scale Pyramid Grid Templates".

## How to run the code
The main function is tst_one.py.  
The method is based on optical flow, therefore you should computed it first and store it in one folder.
And then set the optical flow stored path to the item "optTstPth" in the config.ini.

The provided demo is based on UCSD ped1 dataset, if you want to run the code for your dataset, you should first train the multi-scale maximum pyramid template first, store the template in a suitable path, and set the path to item "tmpPth" in config.ini.

To run the code, you should first modify the config.ini, which has all 
the parameters of the testing. 



