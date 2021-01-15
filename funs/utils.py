import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.cluster import k_means_
import os
from numpy.lib.stride_tricks import as_strided



def getAbnScoreMap(spLabels, abnLabels, abnScore):
    """
    根据得分，获得异常检测的map, the size is the same to the spLabels
    input:
    spLabels ------- superpixel labels
    abnLabels ------ abnormal superpixel labels
    abnScore ------ the corresponding abnormal score
    output:
    shwMap ------ 2d abnormal map
     """
    shwMap = np.zeros(spLabels.shape)
    for i, l in enumerate(abnLabels):
        tmp = spLabels == l
        shwMap[tmp] = abnScore[i]
    return shwMap


def shwAbnSpColImg(img, spLabels, abnLabels):
    """ 在图img中显示异常超像素，该超像素被红色表示
        img -------- original image
        spLabels -------- superpixel labesl of img
        abnLabels -------- abnormal superpixel labels
        output:
        imgCol  ------ image with superpixel colored,
        float type, show using imgCol.astype(np.uint8)
    """
    if len(img.shape) == 2:
        imgCol = np.zeros((img.shape[0], img.shape[1], 3))
        for d in range(3):
            imgCol[:, :, d] = img
    else:
        imgCol = img

    for i, l in enumerate(abnLabels):
        tmp = spLabels == l
        # shwMap[tmp] = abnScore[i]
        for d in range(3):
            if d == 0:
                tmp1 = imgCol[:, :, d]
                tmp1[tmp] = 255
            else:
                tmp1 = imgCol[:, :, d]
                tmp1[tmp] = 0
    return imgCol

def normalizeDt(dtLst, scaler):
    """ normalize the datalist
    dtLst ---- list, one row for one image
    scaler ---- trained data scaler
    """
    # if isinstance(dtLst, list):
    # #     t = np.array(dtLst)
    # #     dtLst = scaler.transform(t)
    # # else:
    #     for i, d in enumerate(dtLst):
    #         # feature for one image
    #         dtLst[i] = scaler.transform(d)
    

    return scaler.transform(dtLst)

def num2str4(n):
    nn = getNumLength(n)
    if nn==1:
        return '000'+str(n)
    if nn==2:
        return '00'+str(n)
    if nn==3:
        return '0'+str(n)
    if nn==4:
        return str(n)
def num2str2(n):
    nn = getNumLength(n)
    if nn==1:
        return '0'+str(n)
    else:
        return str(n)
def num2str3(n):
    nn = getNumLength(n)
    if nn==1:
        return '00'+str(n)
    if nn==2:
        return '0'+str(n)
    if nn==3:
        return str(n)
    
    # if nn==4:
    #     return str(n)
def getNumLength(number):
    Length = 0
    while number != 0:
        Length += 1
        number = number // 10    #关键，整数除法去掉最右边的一位
    return Length   
# def normalize0
# 视频转为图像帧
def getVideoFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    flag, frame = cap.read()
    numFrame+=1
    newPath = os.path.join(svPath, num2str(numFrame)+'.jpg')
    cv2.imencode('.jpg', frame)[1].tofile(newPath)
    
    while flag:
        flag, frame = cap.read()
        if flag:
            numFrame+=1
            print('------videos frames:{}--------'.format(numFrame))
            newPath = os.path.join(svPath, num2str(numFrame)+'.jpg')
            cv2.imencode('.jpg', frame)[1].tofile(newPath)
            cv2.imshow('frame',frame)
            c = cv2.waitKey(1)
            # if c==27:
            #     break
    cap.release()
    cv2.destroyAllWindows()       
    
def shwImgs(rowColums=[1,2], imgList=None, nameList=None):
    if len(imgList) != len(nameList):
        assert('image list shoud equal to the length of name list')
    else:
        # l = len(imgList)
        for index in range(rowColums[0]*rowColums[1]):
            plt.subplot(rowColums[0],rowColums[1],index+1)
            # if index <l:
            plt.imshow(imgList[index],cmap='gray')
            plt.title(nameList[index])
            plt.axis('off')
            
        plt.show()
        
def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

    #光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    #hsv转bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bgr


def magAng2hist( mag, aug, hist_num=8):
    '''
    histogram without normalization
    mag  ----- magnitude map [n, h, w ]
    aug ----- angument map [n,h,w]
    output:
        hist ---- vector of histgram with hist_num
        binNUm ----- pixel number in each direction
    '''
    space = 2*np.pi/hist_num
            # [h,w]
    bins = np.floor(aug/space) 

    hist = np.zeros(hist_num,)
    binNum = np.zeros(hist_num,)
    for i in range(hist_num): 
        indx = bins== i
        ss = np.sum(indx)
        if ss!=0:
            tmp = mag[indx]
            hist[i]=np.sum(tmp)
            binNum[i] = ss
    return hist, binNum
    
    
def pool2d(A, kernel_size, stride, padding=None, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    if padding is not None:
        A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
        
def videoResize(abnMap, sh=120, sw=160):
    t =abnMap.shape[0]
    abn = np.zeros((t, sh, sw))
    for i in range(t):
        m = abnMap[i,...]
        abn[i, ...]=cv2.resize(m, (sw, sh))
        
    return abn

def poolValue(A, kernel_size, stride, padding=None):
        '''
        2D Pooling
    
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            
        '''
        # Padding
        if padding is not None:
            A = np.pad(A, padding, mode='constant')
    
        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                       stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)
    
        return A_w
             
def list2numpy(l):
    '''
    change list to array
    input:
        l ---- list, one element for array
    '''
    t = l[0]
    for i in range(1, len(l)):
        t= np.vstack((t, l[i]))
    return t   


def filterMaps(voteMaps):
    '''
    filtering at the time axis
    [n,h,w]
    '''
    for i in range(voteMaps.shape[2]):
        tmp = voteMaps[...,i]
        tmp = cv2.blur(tmp, (5,5))
        # tmp = cv2.GaussianBlur(tmp, (3,3),0.2)
        voteMaps[...,i]=tmp
    return voteMaps
