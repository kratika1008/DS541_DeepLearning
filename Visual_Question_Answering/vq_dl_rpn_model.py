#make sure you setup tensorflow object detection api in your local system before executing this code.
#follow this link to make your setup: https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
#run this code from the object_detection folder under the research directory.

#load all required libraries
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import json
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

#download faster rcnn model. Here i've downloaded inception v2 model pre-trained on COCO dataset
def download_frcnn_pretrained_model(MODEL_NAME):
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    return PATH_TO_CKPT,PATH_TO_LABELS

#import tensorflow graph
def import_tf_graph(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

#load labels and categories. Not used in our case for now.
def load_labels(PATH_TO_LABELS,NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


#preprocess images and convert into 1D numpy array
def load_and_process_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224,224))
    if np.shape(img)==(224,224,3):
        img_np = np.array(img.getdata()).reshape((224,224,3)).astype(np.uint8)
    else:
        img_np = np.repeat(img.getdata(), 3, -1)
        img_np = img_np.reshape((224,224,3)).astype(np.uint8)
    image_np_expanded = np.expand_dims(img_np, axis=0)
    return image_np_expanded


#get all images for a given path
def get_imgs(main_dir):
    imgs = []
    for filename in os.listdir(main_dir):
        imgs.append(os.path.join(main_dir,filename))
    return imgs


# to get all node names from FRCNN graph
# node_names = [n.name for n in detection_graph.as_graph_def().node]
# with open("FRCNN_Arch_node_names.txt", 'w') as f:
#     for item in node_names:
#         f.write("%s\n" % item)
#     f.close


def get_frcnn_features(detection_graph,imgs):
    all_rpn= []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in imgs:
                image_np_expanded = load_and_process_image(image_path)
                #RPN layer name for FRCNN InceptionV2
                rpn = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4e/concat:0')
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                #get image id from imageName
                img_id = int(img_path[-16:-4])
                (rpn) = sess.run([rpn],feed_dict={image_tensor: image_np_expanded})
                final_vec = np.zeros(rpn[0][:, :, :, 0].shape)
                a,w,h,c = rpn[0].shape
                #flatten rpn vector features
                for i in range(c):
                    final_vec = np.add(final_vec, rpn[0][:, :, :, i])

                final_vec = final_vec.flatten()
                # put imageId at index 0 of the numpy array to align with question features later
                final_vec = np.insert(arr=final_vec,obj=0, values=img_id)
                all_rpn.append(final_vec)

    return all_rpn


if __name__ == "__main__":
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    NUM_CLASSES = 90
    PATH_TO_CKPT,PATH_TO_LABELS = download_frcnn_pretrained_model(MODEL_NAME)
    detection_graph = import_tf_graph(PATH_TO_CKPT)

    #get images from VQA train image dataset (COCO dataset images)
    train_imgs = get_imgs(main_dir='train2014')
    #save rpn features for all train images to numpy file
    np.save('train_img_rpn_features.npy',get_frcnn_features(detection_graph,train_imgs))

    #get images from VQA validation image dataset (COCO dataset images)
    val_imgs = get_imgs(main_dir='val2014')
    #save rpn features for all validation images to numpy file
    np.save('val_img_rpn_features.npy',get_frcnn_features(detection_graph,val_imgs))
