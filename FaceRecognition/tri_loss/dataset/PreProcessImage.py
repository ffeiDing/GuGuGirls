import numpy as np
import cv2
import os
import torchvision.transforms.functional as F
from PIL import  Image
import  random
import math

class PreProcessIm(object):
  def __init__(
      self,
      crop_prob=0,
      crop_ratio=1.0,
      rotate_prob=0,
      rotate_degree=0,
      resize_h_w=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      prng=np.random):
    """
    Args:
      crop_prob: the probability of each image to go through cropping
      crop_ratio: a float. If == 1.0, no cropping.
      resize_h_w: (height, width) after resizing. If `None`, no resizing.
      scale: whether to scale the pixel value by 1/255
      im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
        numpy array with shape [3]
      im_std: (Optionally) divided by image std; `None` or a tuple or list or
        numpy array with shape [3]. Dividing is applied only when subtracting
        mean is applied.
      mirror_type: How image should be mirrored; one of
        [None, 'random', 'always']
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have
        random seed independent from the global one
    """
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.rotate_prob=rotate_prob
    self.rotate_degree=rotate_degree
    self.prng = prng

  def __call__(self, im):
    return self.pre_process_im(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    center_x=im.shape[0]/2
    center_y=im.shape[1]/2
    h_start=int(center_x-new_size[1]/2)
    w_start=int(center_y-new_size[0]/2)
    # print(h_start,w_start,h_start + new_size[1],w_start + new_size[0])
    # print(im.shape)
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])


    # im = np.copy(
    #   im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  @staticmethod
  def rand_rotate(im, degree, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    d = prng.randint(0, degree)
    temp = Image.fromarray(np.uint8(im * 255))
    im=F.rotate(temp, d,expand=True)
    im=np.array(im)

    return im
  @staticmethod
  def rand_erasing(img,sl=0.02, sh=0.2, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):


    for attempt in range(100):
      area = img.shape[0] * img.shape[1]

      target_area = random.uniform(sl, sh) * area
      aspect_ratio = random.uniform(r1, 1 / r1)

      h = int(round(math.sqrt(target_area * aspect_ratio)))
      w = int(round(math.sqrt(target_area / aspect_ratio)))

      if w < img.shape[1] and h < img.shape[0]:
        x1 = random.randint(0, img.shape[0] - h)
        y1 = random.randint(0, img.shape[1] - w)
        if img.shape[2] == 3:
          img[x1:x1 + h, y1:y1 + w,0] = mean[0]
          img[x1:x1 + h, y1:y1 + w,1] = mean[1]
          img[x1:x1 + h, y1:y1 + w,2] = mean[2]
        else:
          img[x1:x1 + h, y1:y1 + w,0] = mean[0]
        return img

    return img

  def pre_process_im(self, im):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Randomly crop a sub-image.
    # showImage = Image.fromarray(np.uint8(im * 255))
    # showImage.show()
    #if ((self.crop_ratio < 1)
    #    and (self.crop_prob > 0)
    #    and (self.prng.uniform() < self.crop_prob)):
    #  h_ratio = self.crop_ratio
    #  w_ratio = self.crop_ratio
    #  crop_h = int(im.shape[0] * h_ratio)
    #  crop_w = int(im.shape[1] * w_ratio)
    #  im = self.rand_crop_im(im, (crop_w, crop_h), prng=self.prng)
    # showImage=Image.fromarray(np.uint8(im*255))
    # showImage.show()


      # im=self.rand_rotate(im,degree=self.rotate_degree,prng=self.prng)
    # showImage = Image.fromarray(np.uint8(im * 255))
    # showImage.show()
    # Resize.
    if (self.resize_h_w is not None) \
        and (self.resize_h_w != (im.shape[0], im.shape[1])):
      #print(self.resize_h_w)
      #print(im.shape)
      im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)#INTER_CUBIC)
      #print(im)
      #im = im.resize(self.resize_h_w, Image.BICUBIC)

    # scaled by 1/255.
    if self.scale:
      im = im / 255.
    #if (self.rotate_prob > 0.1) and (random.uniform(0,1) < self.rotate_prob):
    #  im=self.rand_erasing(im,mean=self.im_mean)
#      showImage = Image.fromarray(np.uint8(im * 255))
#      showImage.save('save_'+str(random.uniform(0,1))+'.jpg')
      


    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored
