
__author__ = 'Jiaqing Zhang '


from PIL import Image
import numpy as np
from scipy import signal
import cv2
import os
def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

def norm(img):
    img = img.squeeze(0)
    norm_img = (img-img.min())/(img.max()-img.min())
    return np.uint8((norm_img)*255)

#####################
#!!!!!!!!!!!note: sample of the one picture
# img_path = 'both.png'
# img = np.array(Image.open(img_path))
# # img = cv2.resize(img, (640, 640))
# rgb_img = img.copy()
# img = np.float32(img) / 255
# # img = cv2.imread('both.png')
# img =np.expand_dims(img,axis=0)
# train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(img, 4)
# train_image_low_8, train_image_high_8 = generateDataWithDifferentFrequencies_3Channel(img, 8)
# train_image_low_12, train_image_high_12 = generateDataWithDifferentFrequencies_3Channel(img, 12)
# train_image_low_16, train_image_high_16 = generateDataWithDifferentFrequencies_3Channel(img, 16)
# train_image_low_20, train_image_high_20 = generateDataWithDifferentFrequencies_3Channel(img, 20)
# train_image_low_24, train_image_high_24 = generateDataWithDifferentFrequencies_3Channel(img, 24)
# train_image_low_28, train_image_high_28 = generateDataWithDifferentFrequencies_3Channel(img, 28)

# img3 = Image.fromarray(np.hstack((rgb_img,norm(train_image_low_4), norm(train_image_high_4),norm(train_image_low_8),norm(train_image_high_8),norm(train_image_low_12),norm(train_image_high_12),norm(train_image_low_16),norm(train_image_high_16),norm(train_image_low_20),norm(train_image_high_20),norm(train_image_low_24),norm(train_image_high_24),norm(train_image_low_28),norm(train_image_high_28))))
# img3.save('all.png')
#########################33

###############
#for the vedai dataset
path = '/home/workshop/dataset/VEDAI/images_ori'
save_path = '/home/workshop/dataset/VEDAI/images_fre'
kernel_size = 100
if not os.path.exists(save_path+'low{}'.format(kernel_size)):
    os.makedirs(save_path+'low{}'.format(kernel_size))
    os.makedirs(save_path+'high{}'.format(kernel_size))
img_list = os.listdir(path)
for img_path in img_list:
    img = np.array(Image.open(path+'/'+img_path))
    img = np.float32(img) / 255
    img =np.expand_dims(img,axis=0)
    train_image_low, train_image_high = generateDataWithDifferentFrequencies_3Channel(img, kernel_size)
    img_low = Image.fromarray(norm(train_image_low))
    img_high = Image.fromarray(norm(train_image_high))
    img_low.save(save_path+'low{}'.format(kernel_size)+'/'+img_path)
    img_high.save(save_path+'high{}'.format(kernel_size)+'/'+img_path)
################

###############
#for the nwpu dataset
# path = '/home/workshop/dataset/VEDAI/images'
# save_path = '/home/workshop/dataset/VEDAI/images_fre'
# kernel_size = 20
# if not os.path.exists(save_path+'low{}'.format(kernel_size)):
#     os.makedirs(save_path+'low{}'.format(kernel_size))
#     os.makedirs(save_path+'high{}'.format(kernel_size))
# img_list = os.listdir(path)
# for img_path in img_list:
#     img = np.array(Image.open(path+'/'+img_path))
#     img = np.float32(img) / 255
#     img =np.expand_dims(img,axis=0)
#     train_image_low, train_image_high = generateDataWithDifferentFrequencies_3Channel(img, kernel_size)
#     img_low = Image.fromarray(norm(train_image_low))
#     img_high = Image.fromarray(norm(train_image_high))
#     img_low.save(save_path+'low{}'.format(kernel_size)+'/'+img_path)
#     img_high.save(save_path+'high{}'.format(kernel_size)+'/'+img_path)
################


# img3 = Image.fromarray(norm(train_image_low_4))
# img3.save('train_image_low_4.png')





