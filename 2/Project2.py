import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy.signal import convolve2d
import cv2
import time


Dx = np.array([[1,0,-1]])
Dy = np.array([[1], [0], [-1]])
box: np.ndarray = np.ones((9,9)) / 81

#Normalizing function
def normalize(im):
    return (im - im.min()) / (im.max() - im.min())


#Naive Convolution
def myconvolve_naive(image, kernel):
    h_kern, w_kern = kernel.shape
    kernel = np.flip(kernel)

    #padding 
    pad_top = (h_kern - 1) // 2
    pad_bottom = (h_kern - 1) - pad_top
    pad_left = (w_kern - 1) // 2
    pad_right = (w_kern - 1) - pad_left
    paddingdims = ((pad_top, pad_bottom), (pad_left, pad_right))
    padded = np.pad(image, paddingdims, mode='constant', constant_values=0)
    h_im, w_im = padded.shape

    output = np.zeros((int(h_im - h_kern + 1), int(w_im - w_kern + 1)))
    h_out, w_out = output.shape

    for row in range(h_out):
        for col in range(w_out):
            outputval = 0
            
            for kernr in range(h_kern):
                for kernc in range(w_kern):
                    imval = padded[row + kernr, col + kernc]
                    kernval = kernel[kernr, kernc]
                    outputval += imval * kernval

            output[row, col] = outputval
    return output

#Optimized convoluton (Inspired by CS 189 HW6.2)
def myconvolve(image, kernel):
    h_kern, w_kern = kernel.shape
    kernel = np.flip(kernel)

    #padding 
    pad_top = (h_kern - 1) // 2
    pad_bottom = (h_kern - 1) - pad_top
    pad_left = (w_kern - 1) // 2
    pad_right = (w_kern - 1) - pad_left
    paddingdims = ((pad_top, pad_bottom), (pad_left, pad_right))
    padded = np.pad(image, paddingdims, mode='constant', constant_values=0)
    h_im, w_im = padded.shape

    output = np.zeros((int(h_im - h_kern + 1), int(w_im - w_kern + 1)))
    h_out, w_out = output.shape

    for r in range(h_out):
        for c in range(w_out):
            rend = r + h_kern
            cend = c + w_kern

            output[r, c] = np.sum(padded[r:rend, c:cend] * kernel)

    return output
    

if __name__ == '__main__':

    print("\n===== PART 1: FUN WITH FILTERS =====\n\n")
    # print("Part 1.1: Convolutions from Scratch")
    # imname = './' + input("Image name (Ex: picture.jpg): ").lower()
    # im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
    # im = sk.img_as_float(im)

    # #Rescaling test image, way too big
    # im = rescale(im, 0.25)

    # plt.imshow(im, cmap='gray')
    # plt.title("Image pre-convolution")
    # plt.axis('off')
    # plt.show()

    # print(f"\nConvolution testing: Naive")
    # now = time.time()
    # print("Box...")
    # box_convolve = myconvolve_naive(im, box)
    # print("Dx...")
    # dx_convolve = myconvolve_naive(im, Dx)
    # print("Dy...")
    # dy_convolve = myconvolve_naive(im, Dy)
    # print(f"Time taken: {(time.time() - now):.2f} seconds")

    # fig, ax = plt.subplots(1, 3, figsize=(8,8))
    # ax[0].imshow(box_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Box')
    # ax[1].imshow(dx_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dx')
    # ax[2].imshow(dy_convolve, cmap='gray')
    # ax[2].set_title('Kernel: Dy')
    # for a in ax:
    #     a.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    # print(f"\nConvolution testing: Optimized")
    # now = time.time()
    # print("Box...")
    # box_convolve = myconvolve(im, box)
    # print("Dx...")
    # dx_convolve = myconvolve(im, Dx)
    # print("Dy...")
    # dy_convolve = myconvolve(im, Dy)
    # print(f"Time taken: {(time.time() - now):.2f} seconds")

    # fig, ax = plt.subplots(1,3, figsize=(8,8))
    # ax[0].imshow(box_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Box')
    # ax[1].imshow(dx_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dx')
    # ax[2].imshow(dy_convolve, cmap='gray')
    # ax[2].set_title('Kernel: Dy')
    # for a in ax:
    #     a.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    # print(f"\nConvolution testing: Library Function (convolve2d)")
    # now = time.time()
    # print("Box...")
    # box_convolve = convolve2d(im, box, mode='same', boundary='fill', fillvalue=0)
    # print("Dx...")
    # dx_convolve = convolve2d(im, Dx, mode='same', boundary='fill', fillvalue=0)
    # print("Dy...")
    # dy_convolve = convolve2d(im, Dy, mode='same', boundary='fill', fillvalue=0)
    # print(f"Time taken: {(time.time() - now):.2f} seconds")

    # fig, ax = plt.subplots(1, 3, figsize=(8,8))
    # ax[0].imshow(box_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Box')
    # ax[1].imshow(dx_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dx')
    # ax[2].imshow(dy_convolve, cmap='gray')
    # ax[2].set_title('Kernel: Dy')
    # for a in ax:
    #     a.axis('off')
    # plt.tight_layout()
    # plt.show()
    


    print("\nPart 1.2: Finite Difference Operator")
    imname = 'cameraman.png'
    im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
    im = sk.img_as_float(im)

    # print("Dx...")
    # dx_convolve = convolve2d(im, Dx, mode='same', boundary='fill', fillvalue=0)
    # print("Dy...")
    # dy_convolve = convolve2d(im, Dy, mode='same', boundary='fill', fillvalue=0)

    # fig, ax = plt.subplots(2, 1, figsize=(8,8))
    # ax[0].imshow(dx_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Dx')
    # ax[1].imshow(dy_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dy')
    # plt.tight_layout()
    # plt.show()

    # gradmagnitude = np.sqrt(dx_convolve ** 2 + dy_convolve ** 2)
    # plt.imshow(gradmagnitude, cmap="gray")
    # plt.title("Gradient Magnitude Image")
    # plt.show()

    # eps = 0.1
    # edgeim = (gradmagnitude > eps).astype(np.float64)

    # plt.imshow(edgeim, cmap="gray")
    # plt.title("Edge Image")
    # plt.show()


    print("\nPart 1.3: Derivative of Gaussian Filter")

    # gaussian = np.outer(cv2.getGaussianKernel(9, 2.0), cv2.getGaussianKernel(9, 2.0, ).T)
    # print("Applying Gaussian...")
    # gauss_convolve = convolve2d(im, gaussian, mode='same', boundary='fill', fillvalue=0)

    # print("Dx...")
    # dx_convolve = convolve2d(gauss_convolve, Dx, mode='same', boundary='fill', fillvalue=0)
    # print("Dy...")
    # dy_convolve = convolve2d(gauss_convolve, Dy, mode='same', boundary='fill', fillvalue=0)

    # fig, ax = plt.subplots(2, 1, figsize=(8,8))
    # ax[0].imshow(dx_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Dx')
    # ax[1].imshow(dy_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dy')
    # plt.tight_layout()
    # plt.show()

    # gradmagnitude = np.sqrt(dx_convolve ** 2 + dy_convolve ** 2)
    # plt.imshow(gradmagnitude, cmap="gray")
    # plt.title("Gradient Magnitude Image")
    # plt.show()

    # eps = 0.1
    # edgeim = (gradmagnitude > eps).astype(np.float64)

    # plt.imshow(edgeim, cmap="gray")
    # plt.title("Edge Image")
    # plt.show()

    # print("\nUsing Derivative of Gaussian...")

    # dogx = convolve2d(gaussian, Dx, mode='same', boundary='fill', fillvalue=0)
    # dogy = convolve2d(gaussian, Dy, mode='same', boundary='fill', fillvalue=0)

    # print("DoGx...")
    # dx_convolve = convolve2d(im, dogx, mode='same', boundary='fill', fillvalue=0)
    # print("D0Gy...")
    # dy_convolve = convolve2d(im, dogy, mode='same', boundary='fill', fillvalue=0)

    # fig, ax = plt.subplots(2, 1, figsize=(8,8))
    # ax[0].imshow(dx_convolve, cmap='gray')
    # ax[0].set_title('Kernel: Dx')
    # ax[1].imshow(dy_convolve, cmap='gray')
    # ax[1].set_title('Kernel: Dy')
    # plt.tight_layout()
    # plt.show()

    # gradmagnitude = np.sqrt(dx_convolve ** 2 + dy_convolve ** 2)
    # plt.imshow(gradmagnitude, cmap="gray")
    # plt.title("Gradient Magnitude Image")
    # plt.show()

    # eps = 0.1
    # edgeim = (gradmagnitude > eps).astype(np.float64)

    # plt.imshow(edgeim, cmap="gray")
    # plt.title("Edge Image")
    # plt.show()



    print("\n\n===== PART 2: FUN WITH FREQUENCIES =====")

    # print("\nPart 2.1: Image Sharpening")
    # imname = 'taj.jpg'
    # im = cv2.imread(imname, cv2.IMREAD_COLOR)
    # im = sk.img_as_float(im)

    # im2name = 'Ladoga.jpg'
    # im2 = cv2.imread(im2name, cv2.IMREAD_COLOR)
    # im2 = sk.img_as_float(im2)


    # def sharpen(im):
    #     sharpened = np.zeros_like(im)
    #     alpha = 2.0

    #     # Color channel logic
    #     for c in range(3):
    #         gaussian = np.outer(cv2.getGaussianKernel(9, 2.0), cv2.getGaussianKernel(9, 2.0, ).T)
    #         smoothed = convolve2d(im[:,:,c], gaussian, mode='same', boundary='fill', fillvalue=0)
    #         details = im[:,:,c] - smoothed
    #         sharpened[:,:,c] = im[:,:,c] + alpha * details
    #     return sharpened
    
    # im1_sharp = sharpen(normalize(im))
    # im2_sharp = sharpen(normalize(im2))

    # fig, ax = plt.subplots(2,2, figsize=(8,4))
    # ax[0,0].imshow(im)
    # ax[0,0].set_title("Original")
    # ax[0,0].axis('off')
    # ax[0,1].imshow(im1_sharp)
    # ax[0,1].set_title(f"Sharpened (alpha = {2.0})")
    # ax[0,1].axis('off')

    # ax[1,0].imshow(im2)
    # ax[1,0].set_title("Original")
    # ax[1,0].axis('off')
    # ax[1,1].imshow(im2_sharp)
    # ax[1,1].set_title(f"Sharpened (alpha = {2.0})")
    # ax[1,1].axis('off')
    # plt.tight_layout()
    # plt.show()

    print("\nPart 2.2: Hybrid Images (See hybrid_python folder)")

    print("\nPart 2.3: Gaussian and Laplacian Stacks")

    def stack(im, levels=5, sigma=2.0, ret='gaussian'):
        gstack = []
        lstack = []
        curr_sigma = sigma

        #Gaussian Stack
        for level in range(levels):
            kernel_size = max(9, int(curr_sigma*6+1))
            if kernel_size % 2 == 0: 
                kernel_size+=1
            
            gaussian = np.outer(cv2.getGaussianKernel(kernel_size, curr_sigma), cv2.getGaussianKernel(kernel_size, curr_sigma).T)

            stack_image = np.zeros_like(im)
            for c in range(3):
                stack_image[:,:,c] = convolve2d(im[:,:,c], gaussian, mode='same', boundary='fill', fillvalue=0)
            gstack.append(stack_image)

            curr_sigma *= 2.0

        #Laplacian Stack
        for level in range(levels - 1):
            lap = gstack[level] - gstack[level + 1]
            lstack.append(lap)
        lstack.append(gstack[-1])

        # Just the gaussian
        if ret.lower() == 'gaussian':
            return gstack
        elif ret.lower() == 'laplacian':
            return lstack
        return gstack, lstack
    
    
    im1name = 'apple.jpeg'
    im1 = cv2.imread(im1name, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im1 = sk.img_as_float(im1)

    im2name = 'orange.jpeg'
    im2 = cv2.imread(im2name, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im2 = sk.img_as_float(im2)

    print("Gaussian and Laplacian stacks...")

    im1 = np.clip(im1, 0, 1)
    im2 = np.clip(im2, 0, 1)

    levels = 5
    agstack, alstack = stack(im1, levels=levels, ret='both')
    ogstack, olstack = stack(im2, levels=levels, ret='both')

    fig, ax = plt.subplots(2, levels, figsize=(20,8))
    
    for i in range(levels):
        ax[0, i].imshow(agstack[i])
        ax[0,i].set_title(f'Gaussian Level {i}')
        ax[1,i].imshow(alstack[i])
        ax[1,i].set_title(f'Laplacian Level {i}')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, levels, figsize=(20,8))
    
    for i in range(levels):
        ax[0, i].imshow(ogstack[i])
        ax[0,i].set_title(f'Gaussian Level {i}')
        ax[1,i].imshow(olstack[i])
        ax[1,i].set_title(f'Laplacian Level {i}')

    plt.tight_layout()
    plt.show()


    print("\nPart 2.4: Multiresolution Blending")

    #Multires
    def multiresblend(im1, im2, mask, levels=5):
        lap1 = stack(im1, levels=levels, ret="laplacian")
        lap2 = stack(im2, levels=levels, ret='laplacian')
        gaussmask = stack(np.stack([mask, mask, mask], axis=2), levels=levels, ret='gaussian')

        combinedstack = []
        for i in range(levels):
            blend = gaussmask[i] * lap1[i] + (1.0 - gaussmask[i]) * lap2[i]
            combinedstack.append(blend)

        blendedim = np.zeros_like(im1)
        for i in combinedstack:
            blendedim += i

        return np.clip(blendedim, 0, 1)
    

    def mask(t, im):
        mask = np.zeros(im.shape[:2])
        if t == 1:
            # Vertical mask
            mask[:, :int(im.shape[1]*0.5)] = 1.0
        elif t == 2:
            #Horizontal mask
            mask[:int(im.shape[0]*0.5), :] = 1.0
        elif t == 3:
            #Ellipsoidal mask
            k, h = (mask.shape[0]//2, mask.shape[1]//2)
            a, b = 120, 80
            x, y = np.ogrid[:mask.shape[0], :mask.shape[1]]
            ellipse = ((x - h)**2 / a**2) + ((y - k**2) / b**2) <= 1.0
            mask[ellipse] = 1.0
        elif t == 4:
            #Vert Striped mask
            # stripe_mask = np.zeros(im4.shape[:2])
            width = 30
            h, w = mask.shape
            for c in range(0, w, width*2):
                mask[:, c:min(c + width, w)] = 1.0
        return mask
    

    # # Orapple
    print('Example 1: Orapple')
    # im1name = 'apple.jpeg'
    # im1 = cv2.imread(im1name, cv2.IMREAD_COLOR)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    # im1 = sk.img_as_float(im1)

    # im2name = 'orange.jpeg'
    # im2 = cv2.imread(im2name, cv2.IMREAD_COLOR)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    # im2 = sk.img_as_float(im2)

    # vmask = mask(1, im1)
    # orapple = multiresblend(im1, im2, vmask, levels=5)
    # fig, ax = plt.subplots(1, 4, figsize=(16,4))
    # ax[0].imshow(im1)
    # ax[0].set_title('Apple')
    # ax[1].imshow(im2)
    # ax[1].set_title('Orange')
    # ax[2].imshow(vmask, cmap='gray')
    # ax[2].set_title('Mask')
    # ax[3].imshow(orapple)
    # ax[3].set_title('Orapple')
    # for a in ax:
    #     a.axis('off')

    # plt.tight_layout()
    # plt.show()

    print('Example 2: Horror Hand')

    
    print('Example 3: Stripes')