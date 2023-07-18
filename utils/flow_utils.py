import os
import cv2
import numpy as np
from struct import pack, unpack


def normalize(arr, t_min, t_max):
    """
    @brief xxx
    @param xxx
    @return xxx
    """
    arr_norm = np.zeros((arr.shape[0],arr.shape[1]), np.float32)
    for x in range(0,len(arr)):
        for y in range(0,len(arr[0])):
            if(arr[x,y] >= t_max):
                arr_norm[x,y] = 1
            else:
                arr_norm[x,y] = round(arr[x,y]/t_max, 2)
    return arr_norm


def readFlowFile(fname):
    '''
    source: https://github.com/youngjung/flow-python
    args
        fname (str)
    return
        flow (numpy array) numpy array of shape (height, width, 2)
    '''

    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(fname)[1]
    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', ('readFlowFile: fname %s should have extension .flo' % fname)

    try:
        fid = open(fname, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', fname)

    tag     = unpack('f', fid.read(4))[0]
    width   = unpack('i', fid.read(4))[0]
    height  = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % fname)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (fname, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (fname, height))

    nBands = 2

    # arrange into matrix form
    flow = np.fromfile(fid, np.float32)
    flow = flow.reshape(height, width, nBands)

    fid.close()

    return flow


def writeFlowFile(img, fname):
    """source: https://github.com/youngjung/flow-python"""
    TAG_STRING = 'PIEH'    # use this when WRITING the file

    ext = os.path.splitext(fname)[1]

    assert len(ext) > 0, ('writeFlowFile: extension required in fname %s' % fname)
    assert ext == '.flo', exit('writeFlowFile: fname %s should have extension ''.flo''', fname)

    height, width, nBands = img.shape

    assert nBands == 2, 'writeFlowFile: image must have two bands'

    try:
        fid = open(fname, 'wb')
    except IOError:
        print('writeFlowFile: could not open %s', fname)
    
    # write the header
    # fid.write(TAG_STRING.encode(encoding='utf-8', errors='strict'))
    # code = unpack('f', bytes(TAG_STRING, 'utf-8'))[0]
    # fid.write(pack('f', code))
    fid.write(bytes(TAG_STRING, 'utf-8'))
    fid.write(pack('i', width))
    fid.write(pack('i', height))

    # arrange into matrix form
    tmp = np.zeros((height, width*nBands), np.float32)

    tmp[:, np.arange(width) * nBands] = img[:, :, 0]
    tmp[:, np.arange(width) * nBands + 1] = np.squeeze(img[:, :, 1])

    fid.write(bytes(tmp))

    fid.close()

    
    
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_to_polar(flow, max_mag=None, min_mag=None, normalize=False, shape=None):

    ## to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    nans = np.isnan(mag)
    if np.any(nans):
        nans = np.where(nans)
        mag[nans] = 0.0
        
    ## magnitude clipping
    if max_mag:
        mag[mag > max_mag] = max_mag


    ## scale to [0, 1]
    if normalize:
        if max_mag:
            mag = mag / max_mag
        else:
            mag = mag / mag.max()
        ang = ang / (2 * np.pi)

    ## drop noise
    if min_mag:
        mag[mag < min_mag] = 0.0
        ang[mag < min_mag] = 0.0
        
    ## put all in a vector
    polar = np.zeros((mag.shape[0], mag.shape[1], 2))
    polar[..., 0] = mag
    polar[..., 1] = ang

    if shape:
        if isinstance(shape, int):
            polar = cv2.resize(polar, (shape, shape), interpolation=cv2.INTER_LINEAR)
        elif isinstance(shape, tuple) or isinstance(shape, list):
            polar = cv2.resize(polar, shape, interpolation=cv2.INTER_LINEAR)
        else:
            raise Exception(f'shape should be an integer or a list [w, h] ! not {type(shape)}')

    return polar


def polar_to_image(polar, max_mag=None, normalized=False):

    """To visualize a polar optical flow"""

    vect = polar.copy()
    
    ## upscale
    if max_mag:
        vect[..., 0] = vect[..., 0] * max_mag
    if normalized:
        vect[..., 1] = vect[..., 1] * 2 * np.pi

    ## to cartesian
    x, y = cv2.polarToCart(vect[..., 0], vect[..., 1])

    ## put all in a vector
    img = np.zeros((vect.shape[0], vect.shape[1], 2))
    img[..., 0] = x
    img[..., 1] = y

    img = flow_to_image(img)
    
    return img



def calc_flow(img1, img2, algo='Farneback'):

    """calculates the optical flow between the two images given the optical flow algorithm."""

    assert algo in ['Farneback', 'DIS', 'Sparse', 'Deep', 'PCA', 'TVL1']

    ## to gray
    if len(img1.shape) == 3 and img1.shape[-1] == 3:
        img1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if algo == 'Farneback':
        flow = cv2.calcOpticalFlowFarneback(im_1, im_2,  None, 0.5, 3, 15, 3, 5, 1.2, 0)

    elif algo == 'DIS':
        inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = inst.calc(im_1, im_2, None)

    elif algo == 'Sparse':
        flow = cv2.optflow.calcOpticalFlowSparseToDense(im_1, im_2, None, 8, 128, 0.05, True, 500.0, 1.5)

    elif algo == 'Deep':
        inst = cv2.optflow.createOptFlow_DeepFlow()
        flow = inst.calc(im_1, im_2, None)

    elif algo == 'PCA':
        inst = cv2.optflow.createOptFlow_PCAFlow()
        flow = inst.calc(im_1, im_2, None)

    else:
        inst = cv2.optflow.createOptFlow_DualTVL1()
        flow = inst.calc(im_1, im_2, None)         

    return flow