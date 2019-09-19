import os
import linecache
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.ndimage as ndimage
import scipy.stats as st
import tensorflow as tf
import pdb; 
def get_line_context(file_path, line_number):
    #获取第line_number行的内容
    return linecache.getline(file_path, line_number).strip()
def load_imagenet(label,batch_shape,root = '../imagenet/ImageNet2012',image_path = 'ILSVRC2012_img_train',train_txt = 'train.txt',index_path = 'index.txt'):
    '''
    load special label's imagenet image and not yield
    batch_shape :shape of minibatch array, i.e. [batch_size, height, width, 3]
    '''
    if int(label)<0:
        label=0
    if int(label)>999:
        label = 999
    images = np.zeros(batch_shape)
    filenames = []
    batch_size = batch_shape[0]
    line = get_line_context(index_path,int(label))
    line = line.rstrip() 
    split_str = line.split()
    begin = int(split_str[1])
    end = int(split_str[2])
    location_file_path = os.path.join(root,train_txt)
    image_path_root = os.path.join(root,image_path)
    idx = 0
    for index in range(begin,end):
        path_str = get_line_context(location_file_path,index)
        path = path_str.rstrip()
        path = path.split()[0]
        image_path = os.path.join(image_path_root,path)
        with tf.gfile.Open(image_path,'rb') as f:
            #pdb.set_trace()
            image = imread(f, mode="RGB").astype(np.float) / 255.0
            #imsave('test.jpg',image)
            #image = np.resize(image,(batch_shape[1],batch_shape[2],batch_shape[3]))
            image = ndimage.zoom(input=image,zoom = [batch_shape[1]/image.shape[0],batch_shape[2]/image.shape[1],1])
            #imsave('testresize.jpg',image)
        images[idx,:,:,:] = image *2.0 -1.0
        idx += 1
        if idx == batch_size:
            return filenames, images
            #yield filenames, images
            images = np.zeros(batch_shape)
            filenames = []
            idx = 0
    if idx > 0:
        return filenames, images
def load_imagenet_batch(label,batch_shape,root = '/media/sdc/datasets/imagenet/ImageNet2012',image_path = 'ILSVRC2012_img_train',train_txt = 'train.txt',index_path = 'index.txt'):
    '''
    load special label's imagenet image using yield,which means the dataloader can be iterated
    batch_shape :shape of minibatch array, i.e. [batch_size, height, width, 3]
    '''
    if int(label)<0:
        label=0
    if int(label)>999:
        label = 999
    images = np.zeros(batch_shape)
    filenames = []
    batch_size = batch_shape[0]
    line = get_line_context(index_path,int(label))
    line = line.rstrip() 
    split_str = line.split()
    begin = int(split_str[1])
    end = int(split_str[2])
    location_file_path = os.path.join(root,train_txt)
    image_path_root = os.path.join(root,image_path)
    idx = 0
    for index in range(begin,end):
        path_str = get_line_context(location_file_path,index)
        path = path_str.rstrip()
        path = path.split()[0]
        image_path = os.path.join(image_path_root,path)
        with tf.gfile.Open(image_path,'rb') as f:
            #pdb.set_trace()
            image = imread(f, mode="RGB").astype(np.float) / 255.0
            image = ndimage.zoom(input=image,zoom = [batch_shape[1]/image.shape[0],batch_shape[2]/image.shape[1],1])
        images[idx,:,:,:] = image *2.0 -1.0
        idx += 1
        if idx == batch_size:
            yield filenames, images
            images = np.zeros(batch_shape)
            filenames = []
            idx = 0
    if idx > 0:
        yield filenames, images




def test():
    
    batch_shape = [32,229,229,3]
    label = 1
    filenames,images = load_imagenet(label,batch_shape)
    print(images.shape)
    imsave(f'{label}batch0.jpg',images[1])
    
test()


