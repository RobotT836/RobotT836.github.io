# Project 1 v2, by Tyler Osbey

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import skimage as sk
from skimage.transform import rescale
from skimage.exposure import adjust_log
import skimage.io as skio

#simple Crop method for preprocessing / removing borders
def crop(im, val):
    return im[val:-val, val:-val]

#Contrast adjustment
def autocontrast(a):
    return adjust_log(a, 1)


# NCC and alignment functions rolled into one
def align(a, b):
    window = 20
    bestdisp = (0,0)
    besterr = -np.inf

    cropval = 20
    acrop = a[cropval: -cropval, cropval:-cropval]
    bcrop = b[cropval: -cropval, cropval:-cropval]
    anorm = acrop - np.mean(acrop)
    bnorm = bcrop - np.mean(bcrop)

    for dx in range(-window, window+1):
        for dy in range(-window, window+1):
            aroll = np.roll(anorm, (dy, dx), axis=(0,1))
            nccerror = np.sum(aroll * bnorm) / (np.sqrt(np.sum(aroll**2)) * np.sqrt(np.sum(bnorm**2)) + 1e-9)

            if nccerror > besterr:
                bestdisp = (dy, dx)
                besterr = nccerror
    
    print(f"Best displacement: {bestdisp}")
    return np.roll(a, bestdisp, axis=(0,1))

# Multiscale ncc adapted for pyramidalign
def multiscalencc(a, b, win):
    bestdisp = (0,0)
    besterr = -np.inf

    #crop images to center, to avoid borders
    cropval = 20
    acrop = a[cropval: -cropval, cropval:-cropval]
    bcrop = b[cropval: -cropval, cropval:-cropval]
    anorm = acrop - np.mean(acrop)
    bnorm = bcrop - np.mean(bcrop)

    #search over the window for best ncc score
    for dx in range(-win, win+1):
        for dy in range(-win, win+1):
            aroll = np.roll(anorm, (dy, dx), axis=(0,1))
            nccerror = np.sum(aroll * bnorm) / (np.sqrt(np.sum(aroll**2)) * np.sqrt(np.sum(bnorm**2)) + 1e-9)

            if nccerror > besterr:
                bestdisp = (dy, dx)
                besterr = nccerror
    
    #print(f"Best displacement: {bestdisp}")
    return bestdisp



def pyramidalign(a, b, levels):
    def pyramidscale(a, b, levels):
        window = 10
        SCALE = 2.0

        #base case
        if levels == 0:
            # Base case: search full window
            win = max(4, window // 2)
            return multiscalencc(a, b, win)
        
        #Downscale
        a_small = rescale(a, 1.0/SCALE, anti_aliasing=True, channel_axis=None)
        b_small = rescale(b, 1.0/SCALE, anti_aliasing=True, channel_axis=None)

        #Coarse displacement
        disp_coarse = pyramidscale(a_small, b_small, levels-1)
        disp_coarse = (int(disp_coarse[0] * SCALE), int(disp_coarse[1] * SCALE))
        a_shifted = np.roll(a, disp_coarse, axis=(0,1))

        #Fine displacement / Scaling correction
        disp_fine = multiscalencc(a_shifted, b, 3)


        bestdisp = (disp_coarse[0] + disp_fine[0], disp_coarse[1] + disp_fine[1])
        print(f"Best Displacement: {bestdisp}")
        return bestdisp
    bestdisp = pyramidscale(a, b, levels)
    return np.roll(a, bestdisp, axis=(0,1))




if __name__ == "__main__":

    imname = './media/' + input("Image name (Ex: cathedral.jpg): ").lower()
    im = skio.imread(imname)
    im = sk.img_as_float(im)

    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    ###PREPROCESSING
    #Normalize brightness via feature scaling (i hope)
    b = (b - b.min()) / (b.max() - b.min())
    g = (g - g.min()) / (g.max() - g.min())
    r = (r - r.min()) / (r.max() - r.min())

    ag = 0
    ar = 0

    now = time.time()

    if imname[len(imname)-3:len(imname)] == 'tif':
        print('TIF detected. Applying multiscale implementation')
        # PREPROCESSING
        b = crop(b, 300)
        g = crop(g, 300)
        r = crop(r, 300)

        print('Applying green channel...')
        ag = pyramidalign(g, b, 4)
        print('Applying red channel...')
        ar = pyramidalign(r, b, 4)
        print('Done')
    else:
        print('JPG detected. Applying single scale implementation')

        # PREPROCESSING
        b = crop(b, 20)
        g = crop(g, 20)
        r = crop(r, 20)

        print('Applying green channel...')
        ag = align(g, b)
        print('Applying red channel...')
        ar = align(r, b)
        print('Done')

    print(f"Time taken: {(time.time() - now):.2f}")

    im_out = np.dstack([ar, ag, b])

    ###POSTPROCESSING
    im_out = autocontrast(im_out)

    fig = plt.figure()
    plt.imshow(im_out)
    plt.title(imname)
    plt.show()

    if input("Save image? [Y/N]: ").lower() == 'y':
        fname = f'./output/out_{str(os.path.basename(imname))[:-4]}.jpg'
        im_out = (np.clip(im_out, 0, 1) * 255).astype(np.uint8)
        skio.imsave(fname, im_out)



