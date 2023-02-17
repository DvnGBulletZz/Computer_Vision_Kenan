from skimage import data, filters, feature, transform
from skimage.util import random_noise
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy
from scipy import ndimage



tform = transform.AffineTransform(scale=(0.5, 1), rotation=0.2, shear = None,translation=(200, 10))
image = data.checkerboard()

transformed_img = transform.warp(image.copy(), tform.inverse)


fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))


axes[0].imshow(image, cmap=plt.cm.gray)
axes[0].set_title('original')

axes[1].imshow(transformed_img, cmap=plt.cm.gray)
axes[1].set_title('tranformed')


plt.tight_layout()
plt.show()



