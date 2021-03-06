import glob
import random
from random import shuffle

random.seed(7)


def get_coord_addr(addr):
    #get coords from file name
    base = addr.split('/')[1]
    base = base.split('.jpg')
    coords_str = base[0].split("_")[1:]
    coords_int = [round(float(coord))for coord in coords_str]
    print("addr=",addr,"coord=",coords_int)
    return coords_int

def resize_crop_pad(img,coord, req_width, req_height):
    result = np.full((req_height,req_width), 255, dtype=np.uint8)
    cropped = img[coord[1]:coord[3],coord[0]:coord[2]]
    while cropped.shape[0] > req_width or cropped.shape[1] > req_height:
        cropped = cv2.resize(cropped, (0, 0), fx=0.5, fy=0.5)
    x = int((result.shape[1] - cropped.shape[1])/2)
    y = int((result.shape[0] - cropped.shape[0])/2)
    result[y:cropped.shape[0]+y, x:cropped.shape[1]+x] = cropped
    return result

train_split = 0.8
image_width = 128
image_height = 128
num_channels = 1
shuffle_data = True

hdf5_path = 'hdf5/shapes_classification.hdf5'
train_path = 'images/*.jpg'
dict_labels = {"Arrow": 0, "Circle": 1, "Rectangle": 2}
addrs = glob.glob(train_path)

labels = []
for addr in addrs:
    for key in dict_labels:
        if key in addr:
            labels.append(dict_labels[key])
print(labels)

if shuffle_data:
    c = list(zip(addrs, labels))  # use zip() to bind the images and labels together
    print(c)
    shuffle(c)
    (addrs, labels) = zip(*c)

train_addrs = addrs[0:int(train_split*len(addrs))]
train_labels = labels[0:int(train_split*len(labels))]

test_addrs = addrs[int(train_split*len(addrs)):]
test_labels = labels[int(train_split*len(labels)):]

##################### second part: create the h5py object #####################
import numpy as np
import h5py

train_shape = (len(train_addrs), image_width, image_height,num_channels)
test_shape = (len(test_addrs), image_width, image_height,num_channels)

# open a hdf5 file and create earrays
f = h5py.File(hdf5_path, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("test_img", test_shape, np.uint8)

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key.
f.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f["train_labels"][...] = train_labels

f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels

######################## third part: write the images #########################
import cv2

# loop over train paths
for i in range(len(train_addrs)):

    if i % 20 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr, 0) #greyscale
    img = resize_crop_pad(img,get_coord_addr(addr),image_width,image_height)
    print("max=", np.max(img), "min=", np.min(img))
    #img = np.expand_dims(img, axis=2)
    #print(img.shape)

    #img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis=2)
    print("final image shape = ",img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB
    #print(img.shape)
    f["train_img"][i] = img

print(f['train_img'].shape)
# loop over test paths
for i in range(len(test_addrs)):

    if i % 20 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr,0)
    img = resize_crop_pad(img,get_coord_addr(addr),image_width,image_height)
    print("max=",np.max(img),"min=",np.min(img))
    #img = np.expand_dims(img,axis=2)
    #print(img.shape)
    #img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis=2)
    #print(img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["test_img"][i] = img
print(f['test_img'].shape)
f.close()