# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:57:41 2020

@author: limax
"""
import funs.utils as utils
import numpy as np
import cv2
# from skimage.segmentation import slic
import os
# from funs.Const import const
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from numba import jit
from scipy import signal

# from skimage.feature import hog
# from skimage.segmentation import felzenszwalb, mark_boundaries
# import datetime

'''
ImgList class, obtaint he file information from path
functions:
    obtImgFileNamesFrmPth------obtain image name list from path
    obtFlderList  ------- obtain folder name list from path
'''


class ImgList():
    def __init__(self, pth):
        '''
        构造函数
        pth   filelist path
        sv_pth    saving path
        '''
        self.pth = pth

    # self.sv_pth = sv_pth
    # @staticmethod
    def _obtImgFileNamesFrmPth(self, pth):
        """ # 从文件夹路径中获得图像的名字列表，返回一个列表
       input:
       pth             folder path
       output:
       file list / namelist has no file type
       """
        flist = []
        namelist = []
        for n in os.listdir(pth):
            fname, ftype = os.path.splitext(n)
            if ftype == '.tif' or ftype == '.jpg' or ftype == '.bmp' or ftype == '.png':
                flist.append(n)
                namelist.append(fname)
        return flist, namelist

    def obtImgFileNames(self):
        """ # 从文件夹路径中获得图像的名字，返回一个列表 """
        flist, namelist = self._obtImgFileNamesFrmPth(self.pth)
        return flist, namelist

    # def obtImgFileNameSave(self):
    def obtFlderList(self, pth):
        """ 获得路径下的文件夹的名字 """
        lists = os.listdir(pth)
        flderList = []
        for f in lists:
            pthNew = os.path.join(pth, f)
            if os.path.isdir(pthNew):
                flderList.append(f)
        return flderList


# *******************************************************
# grid appearance feature gradient
from sklearn.feature_extraction import image


class gridAppPatchFea1video():
    def __init__(self, gx,gy,gt, ptSz=(10,10), maxPtches=None, randSt=0):
        # [n,h,w]
        t = min(gx.shape[0], gy.shape[0], gt.shape[0])
        self.gx = gx[0:t, ...]
        self.gy = gy[0:t, ...]
        self.gt = gt[0:t, ...]
        self.ptSz = ptSz
        self.maxPtches = maxPtches
        self.randSt = randSt
    def extract_pathes(self, g, ptSz=(10,10), step=10):
        '''
        extract the 2-d patches from g [h,w,n] with patch size as ptSz, and space
        :param g:
        :param ptSz: patch size
        :param step: [n, d] n is the sample number, one row for one patch
        :return:
        '''
        imNum = g.shape[2]
        ptches1 = image._extract_patches(g[..., 0], patch_shape=ptSz, extraction_step=step)
        nn=ptches1.shape[0] * ptches1.shape[1]
        ppp = np.empty((nn*imNum, ptSz[0]*ptSz[1]))
        for i in range(imNum):
            ptches1 = image._extract_patches(g[...,i], patch_shape=ptSz, extraction_step=step)
            pt = np.reshape(ptches1, (nn, ptSz[0]*ptSz[1]))
            ppp[i*nn:(i+1)*nn, ...] = pt
        return ppp

    def _obtAppPtchFeaFrmImgs(self, g):
        '''
        obtain apearance feature from images
        :param g:   frames [n, h, w]
        :return:
        '''
        # [n,h,w]
        f = lambda s, step: [s[i:i + step] for i in range(0, len(s) - 1, 1)]
        l = [i for i in range(g.shape[0])]
        lst = f(l, 5)



    # def _obtGrdFeaFrmGrds(self, g, ptSz, maxPtches,randSt):
    #     '''
    #     g is the gradient feature with shape [n,h,w]
    #     :param g: gradient feature
    #     :param ptSz: patch size
    #     :param maxPtches: patches needed to extracted from one frame
    #     :param randSt: random state
    #     :return: pts with one column as one patch
    #     '''
    #     # g is gradient feature [n,h,w]
    #     pts = np.empty((g.shape[0], ptSz*ptSz, maxPtches))
    #     for i in range(g.shape[0]):
    #         p = image.extract_patches_2d(g, ptSz, maxPtches, randSt)
    #         pts[i,...]=p
    #     return pts


    def obtAppFea(self):
        pe = image._extract_patches(self.ptSz, max_patches=self.maxPtches, random_state= self.randSt)


        return px, py, pt



class gridAppGrd1video():
    '''
    given the path of frames, obtain the frame gradients of x, y, t-direction
    output:
    initImgs return the frames in path
    obtTempGrad ---- obtain the t-direction gradient
    obtxyGrad ----- obtain the x and y direction gradients
    '''
    def __init__(self, pth, h=120, w=160):
        '''
        appearance gradient feature x,y,t direction
        :param pth:  video frames path
        :param h: height of resized frame
        :param w: width of the resized frame
        '''
        self.pth = pth
        self.h = h
        self.w = w
        self.initFlg = False
    def initImgs(self):
        self.imgs = self._obtImgsFrmPth()
        self.initFlg = True
        return self.imgs

    def _obtImgsFrmPth(self):
        imgListObj = ImgList(self.pth)
        imgList, nameList = imgListObj.obtImgFileNames()

        imgs = np.zeros((self.h, self.w, len(imgList)), dtype =np.float32)
        cnt = 0
        for f in imgList:
            im1 = cv2.imread(os.path.join(self.pth, f))
            # change to the gray image
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            # remove the noise
            # im1 = cv2.bilateralFilter(im1, d=10, sigmaColor=1, sigmaSpace=15)
            im1 = cv2.GaussianBlur(im1, (3,3), 0.6)

            im1 = cv2.resize(im1, (self.w, self.h))
            im1 = im1/np.max(im1)
            imgs[..., cnt] = im1
            cnt += 1
        return imgs

    def obtTempGrad(self, space=2):
        '''
        get the temporal gradient from path
        :param space: step
        :return:
        '''
        if not self.initFlg:
            self.initImgs()

        f = lambda s, step: [s[i:i + step] for i in range(0, len(s) - 1, space)]
        l = [i for i in range((self.imgs.shape[2]))]
        d = f(l, 2)
        grd = np.zeros((self.h, self.w, (self.imgs.shape[2]) - 1), dtype =np.float32)
        cnt = 0
        for (idx) in d:
            im1 = self.imgs[..., idx[0]]
            im2 = self.imgs[..., idx[1]]

            grd[..., cnt] = im2 - im1
            cnt += 1
        return grd

    def obtxyGrad(self):
        if not self.initFlg:
            self.initImgs()

        grdx = np.zeros((self.h, self.w, (self.imgs.shape[2])), dtype =np.float32)
        grdy = np.zeros((self.h, self.w, (self.imgs.shape[2])), dtype =np.float32)


        for i in range(self.imgs.shape[2]):
            grdx[..., i] = cv2.Sobel(self.imgs[...,i], cv2.CV_32F, 1, 0)
            grdy[..., i] = cv2.Sobel(self.imgs[...,i], cv2.CV_32F, 0, 1)

        
        return grdx, grdy


# =================================================
# '''
# optical flow
# functions:
#     obtOptFlwFrm2Pic ----- obtain optical flow from 2 images
#     obtOptFlwFrm1Pth ----- obtain optical flows from one path
# '''
class GridOptFlow():
    """ initializtion path the continuity frames """

    def __init__(self, pth=None, h=120, w=160):
        # the image folder path
        self.pth = pth
        # width of the resized image
        self.w = w
        # height of the resized image
        self.h = h

    def obtOptFlwFrm2Pic(self, im1, im2, flg='f'):
        '''compute optical flow between im1 and im2
        '''
        # optical flow.
        if flg == 'd':
            inst = cv2.optflow.createOptFlow_DeepFlow()
            deepFlow = inst.calc(im1, im2, None)
        if flg == 'f':
            deepFlow = cv2.calcOpticalFlowFarneback(im1, im2, None,
                                                    pyr_scale=0.5,
                                                    levels=3,
                                                    winsize=10,
                                                    iterations=3,
                                                    poly_n=5,
                                                    poly_sigma=1.1,
                                                    flags=0)
        return deepFlow

    def obtOptFlwFrm1Pth(self, space=1, flg='f'):
        '''obtain the optical flow of one folder
        input:
            space ----- obtain optical flow between i and i+space
        output: list -----containing the flow
        '''
        imgListObj = ImgList(self.pth)
        imgList, nameList = imgListObj.obtImgFileNames()
        f = lambda s, step: [s[i:i + step] for i in range(0, len(s) - 1, space)]
        pList = f(imgList, 2)

        flws = []
        for n in pList:
            print('---------optical flow {} and {}------------'.format(n[0], n[1]))

            im1 = cv2.imread(os.path.join(self.pth, n[0]))
            im2 = cv2.imread(os.path.join(self.pth, n[1]))
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            im1 = cv2.resize(im1, (self.w, self.h))
            im2 = cv2.resize(im2, (self.w, self.h))

            # normalize
            # im1 = im1/np.max(im1)
            # im2 = im2/np.max(im2)
            # gaussian filter, remove noise
            im1 = cv2.GaussianBlur(im1, (3, 3), 0.6)
            im2 = cv2.GaussianBlur(im2, (3, 3), 0.6)
            # im1 = cv2.bilateralFilter(im1, d=10, sigmaColor=100, sigmaSpace=15)
            # im2 = cv2.bilateralFilter(im2, d=10, sigmaColor=100, sigmaSpace=15)

            deepFlow = self.obtOptFlwFrm2Pic(im1, im2, flg)
            flws.append(deepFlow)
        return flws

    def optFlw2Prior(self, flw, hist_num=8):
        '''
        optical flow to prior knowledge
        input: 
            flw ------ optical flow of training data, list of ndarray, [n, h, w,2]
            hist_num ------ orientation numbers
        output:
            maximum of each orientation 
        '''
        flwArray = np.array(flw)
        m, a = cv2.cartToPolar(flwArray[..., 0], flwArray[..., 1])

        space = 2 * np.pi / hist_num
        bins = np.floor(a / space)

        opt_hist = np.zeros((flwArray.shape[1], flwArray.shape[2], hist_num))
        opt_tmp = np.zeros(m.shape)
        for i in range(hist_num):
            indx = bins == i
            opt_tmp[indx] = m[indx]
            opt_hist[:, :, i] = np.max(opt_tmp, axis=0)

        return opt_hist


'''
extract the feature based on optical flow:
    motion feature 
    context motion feature
'''


class AbnFea():
    # optFlw for one video optical flow
    def __init__(self, optFlw):
        # optical flow vectors [n, h, w, 2]
        self.optFlw = optFlw

    def split_by_strides(self, X, kh, kw, s):
        '''
        split x(n, h, w ) into [kh,kw] with step s
        '''

        N, H, W = X.shape
        oh = (H - kh) // s + 1
        ow = (W - kw) // s + 1
        strides = (*X.strides[:-2], X.strides[-2] * s, X.strides[-1] * s,
                   *X.strides[-2:])
        A = np.lib.stride_tricks.as_strided(X, shape=(N, oh, ow, kh, kw), strides=strides)
        return A

    def obtMultScaleHistFea(self, hist_num=8, scales=[1, 2, 3, 4]):

        histDic = {}
        binDic = {}
        # no normalization with bins 3*3
        hist3, bins3 = self.ObtGridOptHistFea(winSize=3,
                                              winStep=3,
                                              # 2020.9.18
                                              tprLen=8,
                                              tprStep=1,
                                              hist_num=hist_num)
        # some other scale hist feature
        for i in scales:
            hist, bin = self.obtHistFeaFrmBaseHist(hist3, bins3, i)
            histDic[i] = hist
            binDic[i] = bin
        return histDic, binDic

    # ======================================================
    # hist feature on optical flow
    # get the histogram feature from optical flows
    # ======================================================
    def ObtGridOptHistFea(self, winSize=3,
                          winStep=3,
                          tprLen=5,
                          tprStep=1,
                          hist_num=2):
        '''
        fast optical flow histogram without normalization
        input:
            winSize ---- kernel size
            winStep ---- kernel stride
            tprLen ---- temporal length
            tprStep ---- temporal sride 
            hist_num ---- histgram orientation number
        output:
            histFea ----[n,nh,nw,hist_num]
        '''
        # compute the magnitude and angles
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])

        poolmags = self.split_by_strides(mag, winSize, winSize, winStep)
        poolangs = self.split_by_strides(ang, winSize, winSize, winStep)

        histFea = np.zeros((*poolmags.shape[:3], hist_num))
        histFeaBins = np.zeros((*poolmags.shape[:3], hist_num))
        for i in range(0, self.optFlw.shape[0] - tprLen + 1, tprStep):
            cubeMags = poolmags[i:tprLen + i, ...]
            cubeAngs = poolangs[i:tprLen + i, ...]

            for ii in range(cubeAngs.shape[1]):
                for jj in range(cubeAngs.shape[2]):
                    cubeMag = cubeMags[:, ii, jj, ...]
                    cubeAng = cubeAngs[:, ii, jj, ...]

                    hist, bins = utils.magAng2hist(cubeMag, cubeAng, hist_num)

                    histFea[i, ii, jj, :] = hist
                    histFeaBins[i, ii, jj, :] = bins

        return histFea, histFeaBins

    # this method is the same to fore-method
    def fastObtOptHistFea(self, winSize=3,
                          winStep=3,
                          tprLen=5,
                          tprStep=1,
                          hist_num=2):
        '''
        fast optical flow histogram 
        input:
            winSize ---- kernel size
            winStep ---- kernel stride
            tprLen ---- temporal length
            tprLen ---- temporal sride 
            hist_num ---- histgram orientation number
        output:
            histFea ----[n,nh,nw,hist_num]
        '''
        # compute the magnitude and angles
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])
        nh = (self.optFlw.shape[1] - winSize) // winStep + 1
        nw = (self.optFlw.shape[2] - winSize) // winStep + 1
        nt = self.optFlw.shape[0]
        poolmags = np.zeros((self.optFlw.shape[0], nh * nw, winSize, winSize))
        poolangs = np.zeros((self.optFlw.shape[0], nh * nw, winSize, winSize))

        for i, (m, a) in enumerate(zip(mag, ang)):
            poolmag = utils.poolValue(m, winSize, winStep)
            poolang = utils.poolValue(a, winSize, winStep)
            poolmags[i, ...] = poolmag
            poolangs[i, ...] = poolang
        histFea = np.zeros((nt, nh, nw, hist_num))
        histFeaBins = np.zeros((nt, nh, nw, hist_num))
        for i in range(0, self.optFlw.shape[0] - tprLen + 1, tprStep):
            cubeMag = poolmags[i:tprLen + i, ...]
            cubeAng = poolangs[i:tprLen + i, ...]

            # cubeMag = cubeMag.reshape(-1, tprLen, winSize, winSize)
            # cubeAng = cubeAng.reshape(-1, tprLen, winSize, winSize)

            hist = np.zeros((cubeMag.shape[1], hist_num))
            bins = np.zeros((cubeMag.shape[1], hist_num))
            for jj in range(cubeMag.shape[1]):
                m = cubeMag[:, jj, ...]
                a = cubeAng[:, jj, ...]
                # histgram 8
                h, b = utils.magAng2hist(m, a, hist_num)
                hist[jj, :] = h
                bins[jj, :] = b
            histFea[i, ...] = hist.reshape((nh, nw, hist_num))
            histFeaBins[i, ...] = bins.reshape((nh, nw, hist_num))
        return histFea, histFeaBins

    def normalizeWithSum(self, hist):
        '''
        normalization using the sumary of each orientation values
        hist ----- histogram [n, w, h, o]
        '''
        # normalization with sum of each histogram
        histSum = np.zeros(hist.shape)
        for i in range(hist.shape[0]):
            t = hist[i, ...]
            s = np.sum(t, axis=2)
            s = np.expand_dims(s, axis=2)

            s = np.tile(s, (1, 1, t.shape[2]))
            t = t / (s + np.finfo(np.float32).eps)
            histSum[i, ...] = t
        return histSum

    def obtHistFeaFrmBaseHist(self, hist, binNums, convSize=2):
        '''
        obtain hist feature from base 3*3 hist feature
            @input:
            histFea ----- 3*3 hist feature (n,w,h,8)
            binNums ----- 3*3 hitting feature numbers in each orientation
            convSize ----- convolution size 1->3*3, 2-->6*6
        @output:
            nhist ---- hist without normalization
            nbins ---- new bins in each orientation
        '''
        # hist = histFea
        kernel = np.ones((convSize, convSize), dtype=np.float32)
        wl, wh, ww, wo = hist.shape[0], hist.shape[1], hist.shape[2], hist.shape[3]
        nwh = wh - convSize + 1
        nww = ww - convSize + 1
        nhist = np.zeros((wl, nwh, nww, wo))
        nbins = np.zeros((wl, nwh, nww, wo))
        # nbins = np.zeros((wl, nwh, nww, wo))
        for i in range(wl):
            hh = hist[i, ...]
            nn = binNums[i, ...]
            for j in range(wo):
                h = hh[..., j]
                n = nn[..., j]
                # convolve2d
                h2 = signal.convolve2d(h, kernel, mode='valid')
                n2 = signal.convolve2d(n, kernel, mode='valid')
                nhist[i, :, :, j] = h2
                nbins[i, :, :, j] = n2
                # if flg == 'ave':

                #     nhist[i, :, :, j] = h2 / (n2 + np.finfo(np.float).eps)
                # elif flg == 'sum':
                #     nhist[i, :, :, j] = h2
        return nhist, nbins

    # 22222222222222222222222222222222222222222222222222222222
    # context feature based on hist feature 
    # 222222222222222222222222222222222222222222222222222222222

    def obtCntFea1FrmHistFea(self, frmHist):
        '''
        context feature with 8-neighbours on one hist map [h,w, bins]
        frmHist ----- hist [h ,w, o]
        '''
        # Chi-Square卡方比较则是，值为0时说明H1= H2，这个时候相似度最高。
        # cv2.HISTCMP_CORREL 直方图相关性比较的值为0，相似度最低
        # cv2.HISTCMP_INTERSECT 十字交叉性 相似则为1，否则小于1
        # HISTCMP_BHATTACHARYYA 巴氏距离的计算结果，其值完全匹配为1，完全不匹配则为0
        method = cv2.HISTCMP_CHISQR

        # padding with the edge pixels
        nhist = np.pad(frmHist, ((1, 1), (1, 1), (0, 0)), mode='edge')

        # neighbourbood size 
        nei_size = 3

        # compute the new height and width
        m = frmHist.shape[0] - nei_size // 2 * 2
        n = frmHist.shape[1] - nei_size // 2 * 2

        cntHist = np.zeros(frmHist.shape, dtype=np.float32)

        for i in range(nei_size // 2, m + nei_size // 2):
            for j in range(nei_size // 2, n + nei_size // 2):
                h0 = nhist[i, j, :]
                cv2.compare

                # upleft
                h1 = nhist[i - 1, j - 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 0] = cv2.compareHist(h1, h0, method=method)

                # up
                h1 = nhist[i - 1, j, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 1] = cv2.compareHist(h1, h0, method=method)

                # upright
                h1 = nhist[i - 1, j + 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 2] = cv2.compareHist(h1, h0, method=method)

                # left
                h1 = nhist[i, j - 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 3] = cv2.compareHist(h1, h0, method=method)

                # right
                h1 = nhist[i, j + 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 4] = cv2.compareHist(h1, h0, method=method)

                # downleft
                h1 = nhist[i + 1, j - 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 5] = cv2.compareHist(h1, h0, method=method)

                # down
                h1 = nhist[i + 1, j, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 6] = cv2.compareHist(h1, h0, method=method)

                # downright
                h1 = nhist[i + 1, j + 1, :]
                cntHist[i - nei_size // 2, j - nei_size // 2, 7] = cv2.compareHist(h1, h0, method=method)

        return cntHist

    def obtCntFeaFrm1FolderHist(self, flderHist):
        '''
        obtain context feature from motion hist feature [n, h, w, bins]
        input :
            flderHist ---- histgram for one folder(one video) [n, h, w, bins]
            '''
        flderHist = np.float32(flderHist)
        cntHist = np.zeros(flderHist.shape, dtype=np.float32)

        for i in range(cntHist.shape[0]):
            frmHist = flderHist[i, ...]
            h = self.obtCntFea1FrmHistFea(frmHist)

            cntHist[i, ...] = h
        return cntHist

    # 333333333333333333333333333333333333333333
    # multi-scale feature 
    # 333333333333333333333333333333333333333333
    def normlizeDicFea(self, histDic, binsDic):
        """
        normalize feature 

        Parameters
        ----------
        hist : [n,h,w,o] hist feature
        bins : [n,h,w,o] bins corresponding to hist 

        Returns
        -------
        histSum : normalize with summary
        histBin : normalize with bins
        """
        histSum = {}
        histBin = {}
        for k, hist in histDic.items():
            bins = binsDic[k]
            histBin[k] = hist / ((bins + np.finfo(np.float).eps))
            histSum[k] = self.normalizeWithSum(hist)

        return histSum, histBin

    def obtMultScaleCntHist(self, histDic):
        """
        obtain multiscale context feature from hist dictionary
        Parameters
        ----------
        histDic : dictionary of different scale hist feature
        Returns
        -------
        cntHistDic : multiscale context feature 
        """

        cntHistDic = {}
        for k, v in histDic.items():
            h = self.obtCntFeaFrm1FolderHist(v)
            cntHistDic[k] = h
        return cntHistDic

    def obtMultScaleHistFrmBase(self, baseHist, bins, convSize=[1, 2, 3, 4]):
        """
        obtain multical hist feature from base 3*3 hist feature, without normalization

        Parameters
        ----------
        baseHist : [n,h,w,o]
            base hist feature with 3*3.
        bins : [n,h,w,o]
            bins corressponding to hist.
        convSize : list, times for 3
            The default is [2, 3, 4].

        Returns
        -------
        histM : dictionary with hist, key is the times
        binM : dictionary with bins, key is the times

        """
        histM = {}
        binM = {}
        for sz in convSize:
            h, m = self.obtHistFeaFrmBaseHist(baseHist, bins, sz)
            histM[(sz)] = h
            binM[(sz)] = m
        return histM, binM


# -----------------------------------------------------------------------
# '''
# anormaly detection with superpixel based on optical flow
# '''
class AbnDetDic():
    '''
    anormaly detection based on multiscale histogram feature
    '''

    def __init__(self, optHistPriorDic, histDic, mask =None):

        # trained hist template stored in dictionary
        self.histTmpDic = optHistPriorDic
        # hist feature 
        self.histFeaDic = histDic
        # mask of active or inactive
        self.mask = mask

    def _calMaxHistDis(self, frmHist, tmp, deta):
        '''
        calculate the distance between one frame histogram and max histogram template
        :param frmHist: array of (h,w,8)
        :param tmp: array of (h,w,8)
        :param deta: scalar
        :return: array of (h,w)
        '''
        d = frmHist - tmp
        d[d < 0] = 0
        d = np.sum(d, axis=2)
        dd = 1 / (1 + np.exp(-np.square(d) / deta))-0.5
        return dd

    def _calPatialHistDis(self, frmHist, tmp, deta, coef=0.8):
        '''
        calculate anomarly map between one frame and histogram template, which is computed from the
        maximum hitogram by coef*tmp
        :param frmHist: one frame histogram
        :param tmp:  maximum histogram template
        :param deta: logistic parameter
        :param coef: coefficient of the comopared template
        :return:
        '''

        tmp = tmp*coef
        return self._calMaxHistDis(frmHist,tmp, deta)


    def abn1PartialFrmHistFea(self, histFea, optFlwPrior, deta=0.01, coef=0.8):
        '''
        anormaly detection between frame and template (coef * template)
        :param histFea: frame histogram [n.h,w]
        :param optFlwPrior: histogram maximum template [h,w]
        :param deta: logistic parameter
        :param coef: coefficient * maximum template
        :return:
        '''
        abnPartialMap = np.zeros(histFea.shape[0:3])
        vLen = histFea.shape[0]
        for t in range(vLen):
            # print('-----anormaly detection frame:{}------'.format(t))
            # ith frame
            hist = histFea[t, ...]

            dd1 = self._calPatialHistDis(hist, optFlwPrior, deta, coef)

            abnPartialMap[t, ...] = dd1

        return abnPartialMap
    def abn1FrmHistFea(self, histFea, optFlwPrior, deta=0.01):
        '''
        anormaly detection from hist feature
        @input:
            histFea ------ histogram in each grid (n, h, w, 8)
            optFlwPrior ----- prior histogram tempalte in each grid (h,w )
        @output:
            abnMap ---- (n, h, w) anormaly probability in each grid 
        '''
        abnMaxMap = np.zeros(histFea.shape[0:3])
        vLen = histFea.shape[0]
        for t in range(vLen):
            # print('-----anormaly detection frame:{}------'.format(t))
            # ith frame
            hist = histFea[t, ...]

            dd = self._calMaxHistDis(hist, optFlwPrior, deta)

            abnMaxMap[t, ...] = dd

        return abnMaxMap

    def abnFrmHistDic(self, deta=0.01, coef =0.8):
        """
        abnormal detection in multiscale hist feature stored in dictionary
        
        Parameters
        ----------
        deta : parameter in anormaly detection the default is const.ABN_DETA.
        coef : maximum template coeffcient for partial anormlay map
        Returns
        -------
        abnMap:     anormaly map dictionary between frames and histogram map
        abnPartialMap: anormaly map dictionary

        """
        abnMap = {}
        abnPartialMap={}
        for k, v in self.histFeaDic.items():
            # print(type(k))
            tmp = self.histTmpDic[k]
            # print(tmp)
            # calcute the distance between the histogram feature and the maximum template
            dm = self.abn1FrmHistFea(v, tmp, deta)
            # compute the distance between the histogram and template *coef
            dp = self.abn1PartialFrmHistFea(v, tmp, deta, coef)
            abnMap[k] = dm
            abnPartialMap[k] = dp
        return abnMap, abnPartialMap

# ==========================================================
# online objective maps
# ==========================================================
class onlineBg():
    def __init__(self, optFlw):
        self.optFlw = optFlw
    def calOnlineBg(self):
        mag, _ = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])
        binMap = np.empty_like(mag)

        for i in range(mag.shape[0]):
            mag1 = mag[i]

            # normalize into 0-255
            cv2.normalize(mag1, mag1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            mag1 = np.uint8(mag1)
            # cv.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
            # tmp = cv2.adaptiveThreshold((mag1), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_OTSU, 25, 10)
            _, dst = cv2.threshold(mag1, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            binMap[i, ...] = dst
        return binMap
# ==========================================================
# combine the anormaly maps containing maximum anormaly map and the
# partial maximum anormaly map
# ===========================================================
class cmb2kindMapDic():
    def __init__(self, maxMap, parMap, glbBg, olBg):
        # ditionary of maximum map
        self.maxMapDic = maxMap
        self.parMapDic = parMap
        # [h,w]
        self.glbBg = glbBg
        # [n,h,w]
        self.olBg = olBg
    def _calMaskInter(self, onlineMask):
        '''
        calculate the intersection between onlineMask and global mask
        :param onlineMask: online mask[n,h,w]
        :return:
        '''
        if self.glbBg is not None:
            tmp = 1-self.glbBg
            tmp = tmp*onlineMask
        else:
            tmp = onlineMask
        return tmp
    # def _calMaskInters(self):
        # for i in range(self.olBg.shape[0]):

    def cmbMaps1(self, maxMap, parMap, olMap):
        '''
        combine one maxMap and partial map using the online map
        :param maxMap: one anormaly detection map with maximum template, (h,w)
        :param parMap: one partial anormaly detection map with maximum * coef template, (h,w)
        :param olMap: one corresponding online map (h,w)
        :return:
        '''
        if not(self.glbBg.shape==maxMap.shape):
            glbBg = cv2.resize(self.glbBg.astype(np.float32), (maxMap.shape[1], maxMap.shape[0]))
            olMap = cv2.resize(olMap, (maxMap.shape[1], maxMap.shape[0]))
            glbBg = glbBg>0
            olMap = olMap>0

        dm = maxMap*glbBg
        inGlbBg = ~glbBg*olMap
        dp = parMap*inGlbBg
        return dm+dp

    def cmbMapsDic(self):
        # first to calculate the intersection map beween global and online maps
        olBg = self._calMaskInter(self.olBg)

        cmbMap={}
        for k, mMap in self.maxMapDic.items():
            scaleMap = np.empty_like(mMap)
            # [n,h,w]
            pMap = self.parMapDic[k]
            for i in range(pMap.shape[0]):
                maxMap = mMap[i]
                parMap = pMap[i]
                scaleMap[i,...]=self.cmbMaps1(maxMap, parMap,olBg[i])

            cmbMap[k]=scaleMap
        return cmbMap



# =========================================================
# processing the template
# =========================================================
class AbnTmpProc():
    def __init__(self, histTmpDic):
        self.histTmpDic = histTmpDic

    def _smooth1Tmp(self, histTmp):
        '''
        smooth the template
        :param tmp: [h,w,o]
        :return:
        '''
        orien = histTmp.shape[2]
        for i in range(orien):
            t = histTmp[..., i]
            # t = cv2.blur(t, (3,3))
            t = cv2.GaussianBlur(t, (3, 3), 1)
            histTmp[..., i] = t
        return histTmp

    def smoothTmpDic(self):
        for k, v in self.histTmpDic.items():
            v = self._smooth1Tmp(v)
            self.histTmpDic[k] = v

        return self.histTmpDic


# ========================================================
# filtering results using background
class AbnMapFilter():
    def __init__(self, abnMap, bg):
        # [n,h,w] detection results
        self.abnMap = abnMap
        self.bg = np.float32(bg)

    def filterAbnMap(self):
        n, *shp = self.abnMap.shape
        # bgt = cv2.resize(self.bg, (shp[1], shp[0]))
        for i in range(n):
            bgt = self.bg[i,...]
            bgt = cv2.resize(bgt, (shp[1], shp[0]))
            self.abnMap[i, ...] = self.abnMap[i, ...] * (bgt > 0)
        return self.abnMap


class AbnDicMapFilter():
    def __init__(self, abnDicMaps, bg):
        self.bg = np.float32(bg)
        self.abnMaps = abnDicMaps

    def filterAbnDicMaps(self):
        for k, v in self.abnMaps.items():
            obj = AbnMapFilter(v, self.bg)
            self.abnMaps[k] = obj.filterAbnMap()
        return self.abnMaps


'''
combine the multiscale results
'''


class AbnCmbMaps():
    def __init__(self, dicMaps):
        # anormlay deteciton maps
        self.abnMapDic = dicMaps

    def cmbMapsPyramid(self):
        return self._cmbDicMapsPyramid(self.abnMapDic)

    def _resize(self, maps, shp):
        '''
        resize 3d maps [n,h,w]
        :param maps:
        :return:
        '''
        m = np.zeros((maps.shape[0], *shp))
        for i in range(maps.shape[0]):
            tmp = maps[i, ...]
            tmp1 = cv2.resize(tmp, (shp[1], shp[0]))
            m[i, ...] = tmp1
        return m

    def _cmbDicMapsPyramid(self, abnMapDic):
        """
        combine multiscale anormaly detection results, according the
        pyramid paper
        Parameters
        ----------
        abnMapDic :multiscale anormaly detection dictionary
        Returns
        -------
        abnMap : combined anormaly detection
        """

        # 1th - scale
        abnMap = abnMapDic[list(abnMapDic.keys())[0]]

        # n, h, w = abnMap.shape[0],abnMap.shape[1], abnMap
        shp = abnMap.shape
        # print(shp, type(shp))
        L = len(abnMapDic)
        abnMap = 1 / (2 ** L) * abnMap

        for k, v in abnMapDic.items():
            if k != list(abnMapDic.keys())[0]:
                l = int(k)
                wght = 1 / (2 ** (L - l))
                if v.shape != shp:
                    # v = np.resize(v, shp)
                    v = self._resize(v, (shp[1], shp[2]))
                v = wght * v
                abnMap += v
        return abnMap

    def cmbMapsByVote(self, vote=2):
        # combine multiscale Maps using vote
        abnMaps = {}
        if len(self.abnMapDic)>1:
            # 1th - scale
            abnMap = self.abnMapDic[1]
            msk = abnMap > 0
            msk = msk + 0
            shp = msk.shape
            for k, v in self.abnMapDic.items():
                if k != 1:
                    if v.shape != shp:
                        v = self._resize(v, (shp[1], shp[2]))
                    tmp = v > 0
                    msk += (tmp + 0)
            inx = msk > vote

            for k, v in self.abnMapDic.items():
                if v.shape != shp:
                    v = self._resize(v, (shp[1], shp[2]))
                tmp = np.zeros(v.shape)
                tmp[inx] = v[inx]
                abnMaps[k] = tmp
        else:
            abnMaps=self.abnMapDic
        return self._cmbDicMapsPyramid(abnMaps)


class AbnDet():
    def __init__(self, optFlwPrior, optTmp, spTmp=None):
        # optical flow prior[h,w, hist_num]
        self.optFlwPrior = optFlwPrior
        # optical flow test [n, h, w]
        self.optTst = optTmp
        # superpixel labels
        self.spTst = spTmp

    def abn1FrmHistFea(self, histFea, optFlwPrior, deta=0.01):
        '''
        anormaly detection from hist feature
        @input:
            histFea ------ histogram in each grid (n, h, w, 8)
            optFlwPrior ----- prior histogram in each grid (h,w )
        @output:
            abnMap ---- (n, h, w) anormaly probability in each grid 
        '''
        abnMap = np.zeros(histFea.shape[0:3])
        vLen = histFea.shape[0]
        for t in range(vLen):
            # print('-----anormaly detection frame:{}------'.format(t))
            # ith frame
            hist = histFea[t, ...]
            # [h,w,8]
            d = hist - optFlwPrior
            d[d < 0] = 0
            d = np.sum(d, axis=2)
            dd = 1 / (1 + np.exp(-np.square(d) / deta)) - 0.5
            abnMap[t, ...] = dd
        return abnMap

    def abn1FrmOptFlw(self, mask=None,
                      HIST_NUM=8,
                      deta=0.01):
        '''
        abnormal detection from one optical flow
        @input:
            optTmp ---optical flow between two images[h,w]
            optFlwPrior ---- training data optical flow prior [HIST_NUM, h,w]
            resMask --- mask of image [h,w]
            HIST_NUM --- orientation number 
            deta ---- distance parameter
        @output:
            anoramly detection result [h, w]
        '''

        m, a = cv2.cartToPolar(self.optTst[..., 0], self.optTst[..., 1])
        abnMap = np.zeros(m.shape)
        space = 2 * np.pi / HIST_NUM
        # [h,w]
        bins = np.floor(a / space)

        # orientation
        if not mask is None:
            bins[mask == 0] = np.inf
            for i in range(HIST_NUM):
                # i-th orientaion
                msk = bins == i
                # prior optical
                optPrior = self.optFlwPrior[..., i]
                optPrior = np.tile(optPrior, [self.optTst.shape[0], 1, 1])
                optPrior = optPrior[msk]
                # test optical flow
                optTst = m[msk]

                d = optTst - optPrior

                # not consider the negative pixels
                d[d < 0] = 0

                dd = 1 / (1 + np.exp(-np.square(d) / deta)) - 0.5
                abnMap[msk] = dd
        else:

            for i in range(HIST_NUM):
                msk = bins == i

                optTst = m[msk]

                optPrior = self.optFlwPrior[..., i]
                optPrior = np.tile(optPrior, [self.optTst.shape[0], 1, 1])
                optPrior = optPrior[msk]
                d = optTst - optPrior
                d[d < 0] = 0

                dd = 1 / (1 + np.exp(-np.square(d) / deta)) - 0.5
                abnMap[msk] = dd

        return abnMap


'''
anormaly detection evaluation ROC
'''

'''
# processing the ground truth
'''


class AbnGt():
    def extGtFrmList(self, gtList, idx, N=200):

        '''
        input:
            gtList ---- ndarray /list
        '''
        if isinstance(gtList, list):
            l = gtList[idx - 1]
        if isinstance(gtList, np.ndarray):
            l = gtList[idx - 1]
        gt = np.zeros(N)
        l = list(np.array(l) - 1)
        gt[l] = 1
        return gt


# anomaly evaluation
class AbnRoc():
    def __init__(self, abnMaps, gt):
        # [n,h,w]
        self.abnMaps = abnMaps
        self.gt = gt

    def _map2MaxScore(self, abnMapList):
        """ change the score list to the max scores for
        drawing the roc
         """
        abnMaxScoreList = []
        for a in abnMapList:
            # one row for one image
            amax = np.max(a)
            abnMaxScoreList.append(amax)
        return abnMaxScoreList

    def abnMapsRoc(self, flg=False):
        '''
        fpr, tpr and draw roc
        :param flg: whether or not draw roc
        :return:
        '''
        maps = self._map2MaxScore(self.abnMaps)
        if not len(maps) == len(self.gt):
            raise AssertionError('length is not equal')
        else:
            fpr, tpr, thresholds = roc_curve(self.gt, maps, pos_label=1)
            roc_auc = auc(fpr, tpr)
            # print('auc==={}'.format(roc_auc))
            if flg:
                plt.plot(fpr, tpr,  # lw=1, alpha=0.3,
                         label='AUC ={}'.format(roc_auc))
                plt.show()
        return fpr, tpr, roc_auc

    def myRoc(self, pred, y):
        '''
        my roc ,return the same fpr and tpr
        :param pred:
        :param y:
        :return:
        '''
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        pred_sort = np.sort(pred)[::-1]  # 从大到小排序
        index = np.argsort(pred)[::-1]  # 从大到小排序
        y_sort = y[index]
        # print(y_sort)
        tpr = []
        fpr = []
        thr = []
        for i, item in enumerate(pred_sort):
            tpr.append(np.sum((y_sort[:i] == 1)) / pos)
            fpr.append(np.sum((y_sort[:i] == 0)) / neg)
            thr.append(item)

        fpr.insert(0, 0)
        fpr.append(1)
        tpr.insert(0, 0)
        tpr.append(1)
        auc1 = auc(fpr, tpr)
        return fpr, tpr, auc1


# **********************************************************
#  show anormaly detection results
# ---------------------------------------
class AbnShw1Map():
    '''
    based on anormaly result, show the detection result on original image
    '''

    def __init__(self, map1, img, thr=0.005):
        # anormaly map [h,w]
        self.abnMap = map1
        # corresponding frame [h1,w1,X]
        if np.ndim(img) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.tile(img, (1, 1, 3))

        self.img = img
        self.thr = thr

    def obtAbnShwMap(self):

        self.abnMap = cv2.resize(self.abnMap, (self.img.shape[1], self.img.shape[0]))
        msk = self.abnMap > self.thr
        self.img[:, :, 0] = msk*255
        # c = [100, 0, 0]
        # for i in range(3):
            
        #     im = self.img[:, :, i]
        #     im[msk] = c[i]
        #     self.img[:, :, i] = im

        return self.img


'''
class AbnShw():
    def __init__(self, abnMaps, imgs):
        # anormaly detection maps dictionary with each value is [n, h, w]
        self.abnMaps = abnMaps
        # [n, h, w]
        self.imgs = imgs

    def shw1ScaleMap(self, abnMap):
        
        # show one scale anormaly detection
        # :param abnMap: [n, h, w]
        # :return:
        
        shwMap = np.zeros(self.imgs.shape,)
        for i in range(abnMap.shape[0]):
            img = self.imgs[i]
            tmap = self.abnMaps[i,...]

            obj = AbnShw1Map(tmap, img)
'''
'''
class of training 
'''


# class AbnStaFeaTrain():
#     def __init__(self):
# ********************************************************
# train histogram feature for one video at one scale (one folder )
class Abn1Scale1VideoHistTrain():
    def __init__(self, histFea, histBin):
        # histFea obtained from optical flow for one video /one folder [n,h,w,o]
        self.histFea = histFea
        self.histBin = histBin
        # store the result
        self.histBinMax = {}

    def trn1ScaleNormMaxHist(self):
        # normalizing with bins
        histBin = self.histFea / ((self.histBin + np.finfo(np.float).eps))
        self.histBinMax = np.max(histBin , axis=0)
        return self.histBinMax

    def trn1ScaleStatHist(self):
        '''
        training histogram statistic feature
        :return:
        '''
        # statistic feature
        h, w, o = self.histBinMax.shape
        shp = (h, w, o + 2)
        histMax = np.zeros(shp)
        histMax[:, :, :o] = self.histBinMax
        if self.histBinMax:
            # [h,w,o]
            ss = np.sum(self.histBinMax, axis=2)
            histMax[:, :, o + 1] = ss
            # ave = np.average(self.histBinMax, axis=2)
            # histMax[:, :, 9] = ave
            std = np.std(self.histBinMax, axis=2)
            histMax[:, :, o + 2] = std
        return histMax


# train multi-scale feature for multiScale
class AbnMultiScaleTrain():
    def __init__(self, histFeaDic, histBinDic):
        # optical flow for one video /one folder， using above the class
        self.histFeaDic = histFeaDic
        # histogram number in each bin corresponding to the self.histFeaDic
        # which is used fot normalization
        self.histBinDic = histBinDic

    def trnDicHistMax(self):
        '''
        obtain the max histogram template from one video hist feature
        :param scales: 1-3*3,2-6*6,3-9*9,4-12*12
        :return: histogram template dictionary [h,w,o]
        '''
        histMaxTmp = {}
        for k, v in self.histFeaDic.items():
            # one scale
            obj = Abn1Scale1VideoHistTrain(v, self.histBinDic[k])
            histMaxTmp[k] = obj.trn1ScaleNormMaxHist()
        return histMaxTmp

    # def update2ScaleHistStat(self, histMax, hist, updtCnt):
    #     '''
    #     update histogram feature, inluding histogram and statistic
    #     :param histMax: [h,w,o+2] hist template
    #     :param hist:  [h,w,o+2] new hist template
    #     :param updtCnt: update times for standard error
    #     :return:
    #     '''
    #
    #     # update hist feature template
    #     histMax[...,:8] =self.update1ScaleHistMax(histMax[...,:8], hist[...,:8])
    #     # update sum
    #     histMax[]

    def update1ScaleHistMax(self, histMax, hist):
        '''
        update histmax template with hist
        :param histMax: old histogram maximum template
        :param hist: other histogram maximum template
        :return:
        '''
        inx = histMax < hist
        histMax[inx] = hist[inx]

        return histMax

    def updateDicHistMax(self, oldHistTmpDic, histTmpDic):
        '''
        update histogram template with other histTmpDic
        :param oldHistTmpDic: old histogram template
        :param histTmpDic:  other histogram template
        :return:
        '''
        for k, v in oldHistTmpDic.items():
            oldHistTmpDic[k] = self.update1ScaleHistMax(oldHistTmpDic[k],
                                                         histTmpDic[k])
        return oldHistTmpDic


class AbnSceneModelTrain():
    def __init__(self, optFlw):
        # [n,h,w,2]
        self.optFlw = optFlw

        n, *shp, o = optFlw.shape
        self.sceneModel = np.zeros(shp)

    def firstTrain(self):
        # only train one video
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])
        self.sceneModel = np.average(mag, axis=0)
        # return self.sceneModel

    def updataTrain(self, optFlw):
        mag, ang = cv2.cartToPolar(optFlw[..., 0], optFlw[..., 1])
        self.sceneModel += np.average(mag)
        return self.sceneModel


class GridTrain():
    def __init__(self, optFlws):
        self.optFlw = optFlws

    @jit
    def trainHistAveSumMax(self, winSize=3,
                           winStep=3,
                           tprLen=5,
                           tprStep=1,
                           hist_num=8):
        '''
        get the prior knowledge
        get the maximum in each position of each window,
        and the normalizaton histogram of each patch
        
        '''
        # magnitude and angles
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])

        height = mag.shape[1]
        width = mag.shape[2]
        # hlfTprLen = int(np.floor(tprLen/2))
        # hlfWinSz = int(np.floor(winSize/2))
        frms = len(mag)

        # the prior matrix
        hn = len(range(0, height - winSize + 1, winStep))
        wn = len(range(0, width - winSize + 1, winStep))

        histAvePrior = np.zeros((hn, wn, hist_num))
        histSumPrior = np.zeros((hn, wn, hist_num))

        for ii, row in enumerate(range(0, height - winSize + 1, winStep)):
            for jj, col in enumerate(range(0, width - winSize + 1, winStep)):

                histAveMax = np.zeros(hist_num, )
                histSumMax = np.zeros(hist_num, )
                for k, frmId in enumerate(range(0, frms - tprLen + 1, tprStep)):
                    magCube = mag[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]

                    angCube = ang[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]
                    hist, binNum = utils.magAng2hist(magCube, angCube, hist_num)

                    # normalize 
                    hist_norm = hist / (binNum + np.finfo(np.float32).eps)

                    # get the maximum in each orientation
                    indx = hist_norm > histAveMax
                    histAveMax[indx] = hist_norm[indx]

                    # hist = hist * binNum
                    indx = hist > histSumMax
                    histSumMax[indx] = hist[indx]
                    # normalize

                    # print(histMax)
                histAvePrior[ii, jj, :] = histAveMax
                histSumPrior[ii, jj, :] = histSumMax

        return histAvePrior, histSumPrior

    @jit
    def trainHistAveMax(self, winSize=3,
                        winStep=3,
                        tprLen=5,
                        tprStep=1,
                        hist_num=2):
        '''
        get the prior knowledge
        get the maximum in each position of each window
        
        '''
        # magnitude and angles
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])

        height = mag.shape[1]
        width = mag.shape[2]
        # hlfTprLen = int(np.floor(tprLen/2))
        # hlfWinSz = int(np.floor(winSize/2))
        frms = len(mag)

        # the prior matrix
        hn = len(range(0, height - winSize + 1, winStep))
        wn = len(range(0, width - winSize + 1, winStep))

        histPrior = np.zeros((hn, wn, hist_num))

        for ii, row in enumerate(range(0, height - winSize + 1, winStep)):
            for jj, col in enumerate(range(0, width - winSize + 1, winStep)):

                histMax = np.zeros(hist_num, )
                for k, frmId in enumerate(range(0, frms - tprLen + 1, tprStep)):
                    magCube = mag[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]

                    angCube = ang[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]

                    # hist without normalization
                    hist, binNum = utils.magAng2hist(magCube, angCube, hist_num)

                    # normalize ----average with the summary bins in each orienation
                    hist = hist / (binNum + np.finfo(np.float32).eps)

                    # get the maximum in each orientation
                    indx = hist > histMax
                    histMax[indx] = hist[indx]
                    # normalize

                    # print(histMax)
                histPrior[ii, jj, :] = histMax

        return histPrior

    def trainHistMax(self, winSize=3,
                     winStep=3,
                     tprLen=5,
                     tprStep=1,
                     hist_num=2):
        '''
        get the prior knowledge
        get the maximum in each position of each window patch
        
        '''
        # magnitude and angles
        mag, ang = cv2.cartToPolar(self.optFlw[..., 0], self.optFlw[..., 1])

        height = mag.shape[1]
        width = mag.shape[2]
        # hlfTprLen = int(np.floor(tprLen/2))
        # hlfWinSz = int(np.floor(winSize/2))
        frms = len(mag)

        # the prior matrix
        hn = len(range(0, height - winSize + 1, winStep))
        wn = len(range(0, width - winSize + 1, winStep))

        histPrior = np.zeros((hn, wn, hist_num))

        for ii, row in enumerate(range(0, height - winSize + 1, winStep)):
            for jj, col in enumerate(range(0, width - winSize + 1, winStep)):
                histMax = np.zeros(hist_num, )
                for k, frmId in enumerate(range(0, frms - tprLen + 1, tprStep)):
                    magCube = mag[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]

                    angCube = ang[frmId:frmId + tprLen,
                              row:row + winSize,
                              col:col + winSize]
                    hist, binNum = utils.magAng2hist(magCube, angCube, hist_num)

                    # hist = hist * binNum
                    # normalize 
                    # hist = hist/(binNum+np.finfo(np.float32).eps)

                    # get the maximum in each orientation
                    indx = hist > histMax
                    histMax[indx] = hist[indx]
                    # normalize

                    # print(histMax)
                histPrior[ii, jj, :] = histMax

        return histPrior
