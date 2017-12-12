import numpy as np
import matplotlib.pyplot as plt
#import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset


fname = "images/download.jpg"
real_image = ndimage.imread(fname)

#http://www.scipy-lectures.org/advanced/image_processing/


image = np.array(real_image)

#im_noise = image + 0.1 * np.random.randn(*image.shape)
#im_med = ndimage.median_filter(im_noise, 3)
#real_image.transform(image.size, Image.AFFINE, None, resample=Image.BICUBIC)

#image = im_med


#image = scipy.misc.imread(image)
#img = Image.fromarray(image, 'RGB')
print(image.shape)
plt.imshow(image)
plt.show()
#plt.show(image)