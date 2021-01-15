#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rdConfig.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/9/18 9:31   gxrao      1.0         None
'''

# import lib
import configparser
import numpy as np

class config():
    def __init__(self, pth='config.ini', section='ucsdPed1'):
        
        self.sec = section
        self.cf = configparser.ConfigParser()
        self.cf.read(pth)

    # original image path
    def getImgTrnPth(self):
        return self.cf.get(self.sec, "imgPthTrain")

    def getImgTstPth(self):
        return self.cf.get(self.sec, "imgPthTest")

    # ground truth path
    def getGtPth(self):
        return self.cf.get(self.sec, "gtPth")

    # opitcal flow path
    def getOptTrnPth(self):
        return self.cf.get(self.sec, "optTrnPth")
    def getOptTstPth(self):
        return self.cf.get(self.sec, "optTstPth")

    # tempalte path
    def getTmpPth(self):
        return self.cf.get(self.sec, "tmpPth")

    # scene model path
    def getBgPth(self):
        return self.cf.get(self.sec, "bgPth")

    def getFrmResPth(self):
        return self.cf.get(self.sec, 'resFrmPth')
    
    # get the scene model threshold
    def getBgThr(self):
        return np.int(self.cf.get(self.sec, 'bgThr'))
    
    # get histogram directions number
    def getHistDic(self):
        histDic={}
        histDic['histN']=np.int(self.cf.get(self.sec, 'histN'))
        histDic['winSize']=np.int(self.cf.get(self.sec, 'winSize'))
        histDic['winStep']=np.int(self.cf.get(self.sec, 'winStep'))
        histDic['tprLen']=np.int(self.cf.get(self.sec, 'tprLen'))
        histDic['tprStep']=np.int(self.cf.get(self.sec, 'tprStep'))
        return histDic

    # get the weakfactor
    def getWeakFactors(self): 
        
        return np.float(self.cf.get(self.sec, 'WeakFactor'))
    # get image type
    def getImgType(self): 
        return (self.cf.get(self.sec, 'imgType'))
    # get deta param in abnormal detectoin 
    def getDeta(self): 
        return np.float(self.cf.get(self.sec, 'deta'))