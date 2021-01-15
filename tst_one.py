# -*- coding: utf-8 -*-
# only used for testing
import cv2
from pathlib import Path
import os
import numpy as np
import funs.utils as utils
from funs.grdOpt import AbnFea, AbnDetDic, AbnMapFilter,AbnRoc, AbnGt, GridOptFlow, AbnShw1Map, AbnCmbMaps, onlineBg, cmb2kindMapDic
import matplotlib.pyplot as plt
from funs.grdOpt import AbnTmpProc

from funs.rdConfig import config
cf = config(pth='config.ini', section='ucsdPed1')
# optical flow path
optTrnPth = cf.getOptTrnPth()
optPth = cf.getOptTstPth()
# ground truth path
gtPth = cf.getGtPth()
# templates
tmpPth = cf.getTmpPth()
# scene model
bgPth = cf.getBgPth()
# scene model threshold
snThr = cf.getBgThr()
# histogram number 
histDic = cf.getHistDic()
weakFactors = cf.getWeakFactors()
deta = cf.getDeta()
# image typpe
imgType = cf.getImgType()


imgPth = cf.getImgTstPth()
# bgPth = 'data/ped1Bg.npy'
bg = np.load(bgPth)

if bg is not None:
    if len(np.unique(bg)>2):
        # bg=bg.astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # bg = clahe.apply(bg)
        # _, bg = cv2.threshold(bg, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bg = bg>snThr


IDX = 15


name = Path('Test'+utils.num2str3(IDX)+'.npy')
# name = Path(str(IDX)+'.npy')

optName = Path(optPth).joinpath(name)
# load the tempaltes
histBinTmpDicTmp=np.load(tmpPth, allow_pickle=True).item()

# opt = np.array(opt)
opt= np.load(optName)
# mag, ang = cv2.cartToPolar(opt[..., 0], opt[..., 1])


abnFeaObj = AbnFea(opt)
hist3, bin3 = abnFeaObj.ObtGridOptHistFea(winSize=histDic['winSize'], winStep=histDic['winStep'],
                                              tprLen=histDic['tprLen'], tprStep=histDic['tprStep'],
                                              hist_num=histDic['histN'])
# 通过3*3获得 其他尺度的特征
histMDic, binMDic = abnFeaObj.obtMultScaleHistFrmBase(hist3, bin3)
# normalize
histNsumFea, histNbinFea = abnFeaObj.normlizeDicFea(histMDic, binMDic)

# smooth the templates 
# for k,v in histBinTmpDicTmp.items():
#     for i in range(v.shape[2]):
#         tmp = cv2.GaussianBlur(v[...,i], (3,3), 0.5)
#         tmp = np.max(v)
#         v[...,i]=tmp
#     histBinTmpDicTmp[k]=v

abnDetObj = AbnDetDic(histBinTmpDicTmp, histNbinFea, bg)
# anormaly detection on multiscale

abnBinMaps, abnParMaps = abnDetObj.abnFrmHistDic(deta = deta, coef=weakFactors)

# ============================================
# calculate online background
# =============================================
olObj =onlineBg(opt)
olBg = olObj.calOnlineBg()

# =============================================
# =============================================
cmb2MapsObj = cmb2kindMapDic(abnBinMaps, abnParMaps, bg, olBg)
abnBinMaps = cmb2MapsObj.cmbMapsDic()


# combine multiscale results
cmbObj = AbnCmbMaps(abnBinMaps)
pramidMaps = cmbObj.cmbMapsPyramid()
voteMaps = cmbObj.cmbMapsByVote(vote=1)

# filtering result
flterObj = AbnMapFilter(voteMaps, olBg)
voteMaps = flterObj.filterAbnMap()
flterObj = AbnMapFilter(pramidMaps, olBg)
pramidMaps = flterObj.filterAbnMap()

# filtering at temporal axis
voteMaps = utils.filterMaps(voteMaps)
pramidMaps =utils.filterMaps(pramidMaps)  

# %%
'''
===============================================
显示检测结果
==============================================
'''
selId=138
# imgPth1 = imgPth+utils.num2str2(IDX)
imgPth1 = imgPth+'Test'+utils.num2str3(IDX)
# fPth = imgPth1
# fPth = Path(imgPth1).joinpath(Path(utils.num2str4(selId)+'.jpg'))
img = cv2.imread(os.path.join(imgPth1, utils.num2str3(selId)+imgType))
# img = cv2.imread(str(fPth))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(160,120))
imglist=[]
namelist=[]
for i in range(4):
    shwMap = abnBinMaps[i+1][selId,...]
    shwObj = AbnShw1Map(shwMap, img)
    tmp = shwObj.obtAbnShwMap()
    imglist.append(tmp)
    namelist.append('scale:'+str(i))

shwObj = AbnShw1Map(pramidMaps[selId,...], img)
imglist.append(shwObj.obtAbnShwMap())
shwObj = AbnShw1Map(voteMaps[selId,...], img)
imglist.append(shwObj.obtAbnShwMap())

namelist.append('pramid')
namelist.append('vote')
utils.shwImgs([2,3], imgList=imglist, nameList= namelist)
# %%
# ============================================
# evaluate result
gtFile = list(np.load(gtPth, allow_pickle=True))
gtObj = AbnGt()
#
gt = gtObj.extGtFrmList(gtList=gtFile, idx=IDX, N= len(opt)+1)
gt = gt[:len(voteMaps)]
#
evlObj = AbnRoc(voteMaps, gt)
fpr, tpr, auc = evlObj.abnMapsRoc()
print('---vote auc{}'.format(auc))

evlObj = AbnRoc(pramidMaps, gt)
fpr, tpr, auc = evlObj.abnMapsRoc()
print('---pramidMaps auc{}'.format(auc))

# fpr, tpr, auc = evlObj.myRoc(abnBinMaps,gt)
# %%


