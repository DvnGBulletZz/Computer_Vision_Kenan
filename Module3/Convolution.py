from skimage import data, filters, feature
from skimage.util import random_noise
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy
from scipy import ndimage






image = data.camera()

# viewer = ImageViewer(image)
# viewer.show()


mask1=[[1,0],[0, -1]]    #roberts
mask11=[[0,1],[-1,0]]    #roberts

mask2=[[1,0,-1],[1,0,-1],[1,0,-1]] #prewitt
mask21=[[1,1,1],[0,0,0],[-1,-1,-1]] #prewitt

mask3=[[-1,0,1],[-2,0,2],[-1,0,1]] #sobel
mask31=[[1,2,1],[0,0,0],[-1,-2,-1]] #sobel



skimage_response = filters.gaussian(image, 0.1, channel_axis=True)
roberts1 = ndimage.convolve(skimage_response, mask1)
roberts2 = ndimage.convolve(skimage_response , mask11)

# print(newimage11)

robert_edge = np.sqrt(np.square(roberts1) + np.square(roberts2))
robert_edge *= 255.0 / robert_edge.max()



prewitt1 = ndimage.convolve(skimage_response, mask2)
prewitt2 = ndimage.convolve(skimage_response, mask21)

prewitt_edge = np.sqrt(np.square(prewitt1) + np.square(prewitt2))
prewitt_edge *= 255.0 / prewitt_edge.max()

sobel1 = ndimage.convolve(skimage_response, mask3)
sobel2 = ndimage.convolve(skimage_response, mask31)

sobel_edge = np.sqrt(np.square(sobel1) + np.square(sobel2))
sobel_edge *= 255.0 / sobel_edge.max()

edge_roberts = filters.roberts(image)
edge_sobel   = filters.sobel(image)
edge_prewitt = filters.prewitt(image)


# cannyimg = ndimage.rotate(image, 15, mode='constant')
cannyimg = ndimage.gaussian_filter(image, 4)
cannyimg = random_noise(cannyimg, mode='speckle', mean=0.1)

edge_canny = feature.canny(cannyimg, sigma=2)


fig, axes = plt.subplots(ncols=7, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

axes[2].imshow(edge_prewitt, cmap=plt.cm.gray)
axes[2].set_title('prewitt Edge Detection')

axes[3].imshow(edge_canny, cmap=plt.cm.gray)
axes[3].set_title('Canny Edge Detection')


axes[4].imshow(robert_edge, cmap=plt.cm.gray)
axes[4].set_title('own mask 1')

axes[5].imshow(prewitt_edge, cmap=plt.cm.gray)
axes[5].set_title('own mask 2')

axes[6].imshow(sobel_edge, cmap=plt.cm.gray)
axes[6].set_title('own mask 3')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

# mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
