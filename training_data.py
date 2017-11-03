import os
import numpy as np
from scipy.misc import imread, imresize

count = -2
train_labels = []
train_images = []
for root, dirs, files in os.walk("Tomato"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))        
    count = count + 1
    for file in files:
        img = imresize(imread("Tomato/"+os.path.basename(root)+'/'+file, mode='RGB'), (60, 60)).astype(np.float32)
        #img[:, :, 0] -= 123.68
        #img[:, :, 1] -= 116.779
        #img[:, :, 2] -= 103.939
        #img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        #img = img.transpose((2, 0, 1))
        #img = np.expand_dims(img, axis=0)
        train_images.append(img)
        train_labels.append(count)
        print(len(path) * '---', file)
print(len(train_images))
print(len(train_labels))
#print(train_labels)
np.save('train_images_lenet.npy',np.array(train_images))
np.save('train_labels_lenet.npy',np.array(train_labels))