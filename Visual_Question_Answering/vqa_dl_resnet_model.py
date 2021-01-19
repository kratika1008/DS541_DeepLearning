# importing libraries
from keras.models import Model
from PIL import Image
from keras.applications.resnet50 import ResNet50
import numpy as np
import os
import glob


#function to remove last layer (softmax layer) of resnet50
def pop(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False
    return model


#get all images for a given path
def get_imgs(main_dir):
    imgs = []
    for filename in os.listdir(main_dir):
        imgs.append(os.path.join(main_dir,filename))
    return imgs


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


# process all train images to generate resnet50 features
def get_resnet_img_features(imgs):
    all_image_features= []
    for img_path in imgs:
        img_np_expanded = load_and_process_image(img_path)
        #get image id from imageName
        img_id = int(img_path[-16:-4])
        image_features = res_model.predict(img_np_expanded)[0]
        # put imageId at index 0 of the numpy array to align with question features later
        image_features = np.insert(arr=image_features,obj=0, values=img_id)
        all_image_features.append(image_features)
    return all_image_features


if __name__ == "__main__":
    #load pretrained resnet50 model trained with imagenet weights
    res_model = ResNet50(weights='imagenet')
    # remove last layer (softmax layer) of the model
    res_model = pop(res_model)

    #get images from VQA train image dataset (COCO dataset images)
    train_imgs = get_imgs(main_dir='train2014')
    #save resnet features for all train images to numpy file
    np.save('train_img_resnet_features.npy',get_resnet_img_features(train_imgs))

    #get images from VQA validation image dataset (COCO dataset images)
    val_imgs = get_imgs(main_dir='val2014')
    #save resnet features for all validation images to numpy file
    np.save('val_img_resnet_features.npy',get_resnet_img_features(val_imgs))
