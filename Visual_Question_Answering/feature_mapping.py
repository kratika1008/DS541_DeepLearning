#import libraries
import numpy as np
import pandas as pd
from keras.utils import np_utils
import json

#load question to index file
def load_ques_to_index():
    f = open('Question_i_9.json')
    ques_index = json.load(f)
    return ques_index

#load answer to index file
def load_answer_to_index():
    f = open('ansvocab_i_9.json')
    ans_index = json.load(f)
    return ans_index

def load_ques_img_mapping():
    preprocessed_data = pd.read_csv('preprocessed1_whole.csv')
    preprocessed_data = preprocessed_data.drop(columns=['processed_tokens','final_question'],axis=1)
    return preprocessed_data

def load_ques_ans_mapping():
    ques_ans_df = pd.read_csv('Question_Answer_9.csv')
    ques_ans_df = ques_ans_df.drop(['Unnamed: 0'],axis=1)
    ques_ans_df.columns = ['Ques '+str(col)  for col in ques_ans_df.columns]
    ques_ans_df = ques_ans_df.rename(columns={'Ques Question_id': 'Question_id', 'Ques Answer':'Answer'})
    return ques_ans_df

def get_img_cnn_features():
    cnn_img_all = pd.DataFrame()
    try:
        img_cnn_train = np.load('image_features_resnet.npy')
        imdir='COCO_%s_%012d.jpg'
        subtype = 'train2014'
        image_filenames=img_cnn_train[:,0]
        image_filenames=[imdir%(subtype,i) for i in image_filenames]
        im_df=pd.DataFrame(img_cnn_train[:,1:])
        im_df['image_filenames']=image_filenames

        img_cnn_val = np.load('val_image_features_resnet.npy')
        subtype = 'val2014'
        image_filenames=img_cnn_val[:,0]
        image_filenames=[imdir%(subtype,i) for i in image_filenames]
        im_val_df=pd.DataFrame(img_cnn_val[:,1:])
        im_val_df['image_filenames']=image_filenames
        cnn_img_all = im_df.append(im_val_df)
        cnn_img_all.columns = ['CNN_Img '+str(col)  for col in cnn_img_all.columns]
        cnn_img_all = cnn_img_all.rename(columns={'CNN_Img image_filenames': 'image_filenames'})
    except:
        del img_cnn_train,image_filenames,im_df,img_cnn_val,im_val_df

    return cnn_img_all

def get_img_rpn_features():
    cnn_img_all = pd.DataFrame()
    try:
        img_rpn_train = np.load('train_img_rpn_features.npy')
        imdir='COCO_%s_%012d.jpg'
        subtype = 'train2014'
        image_filenames=img_rpn_train[:,0]
        image_filenames=[imdir%(subtype,i) for i in image_filenames]
        im_df=pd.DataFrame(img_rpn_train[:,1:])
        im_df['image_filenames']=image_filenames

        img_rpn_val = np.load('val_img_rpn_features.npy')
        subtype = 'val2014'
        image_filenames=img_rpn_val[:,0]
        image_filenames=[imdir%(subtype,i) for i in image_filenames]
        im_val_df=pd.DataFrame(img_rpn_val[:,1:])
        im_val_df['image_filenames']=image_filenames

        rpn_img_all = im_df.append(im_val_df)
        rpn_img_all.columns = ['RPN_Img '+str(col)  for col in rpn_img_all.columns]
        rpn_img_all = rpn_img_all.rename(columns={'RPN_Img image_filenames': 'image_filenames'})
    except:
        del img_rpn_train,image_filenames,im_df,img_rpn_val,im_val_df

    return rpn_img_all

def generate_ques_embedding(preprocessed_data,ques_ans_df):
    ques_df = preprocessed_data.merge(ques_ans_df,left_on='ques_id',right_on='Question_id')
    return ques_df

def generate_ques_ans_cnn_df(ques_df,cnn_img_all):
    ques_ans_cnn_df = ques_df.merge(cnn_img_all,left_on='img_path',right_on='image_filenames')
    return ques_ans_cnn_df

def generate_ques_ans_rpn_df(ques_df,rpn_img_all):
    ques_ans_rpn_df = ques_df.merge(rpn_img_all,left_on='img_path',right_on='image_filenames')
    return ques_ans_rpn_df

def get_answer_encoded(ques_ans_cnn_df):
    ans_encoded_train = pd.DataFrame()
    ans_encoded_val = pd.DataFrame()
    try:
        all_ans = ques_ans_rpn_df[['Answer','dataset']]
        all_ans_encoded = np_utils.to_categorical(all_ans[['Answer']].to_numpy())
        all_ans_encoded_df = all_ans_encoded_df.join(all_ans['dataset'])
        ans_encoded_train = all_ans_encoded_df[all_ans_encoded_df.dataset=='train']
        ans_encoded_val = all_ans_encoded_df[all_ans_encoded_df.dataset=='val']

        ans_encoded_val = ans_encoded_val.drop(['dataset'],axis=1)
        ans_encoded_val = ans_encoded_val.to_numpy()

        ans_encoded_train = ans_encoded_train.drop(['dataset'],axis=1)
        ans_encoded_train = ans_encoded_train.to_numpy()
    except:
        del all_ans,all_ans_encoded,all_ans_encoded_df
    return ans_encoded_train,ans_encoded_val

def get_ques(ques_ans_cnn_df):
    spike_cols = [col for col in ques_ans_cnn_df.columns if 'Ques' in col]
    spike_cols = spike_cols.remove('Question_id')
    ques_train = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='train'][spike_cols].to_numpy()
    ques_val = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='val'][spike_cols].to_numpy()
    return ques_train,ques_val

def get_cnn_imgs(ques_ans_cnn_df):
    spike_cols = [col for col in ques_ans_cnn_df.columns if 'CNN_Img' in col]
    cnn_train = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='train'][spike_cols].to_numpy()
    cnn_val = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='val'][spike_cols].to_numpy()
    return cnn_train,cnn_val

def get_rpn_imgs(ques_ans_cnn_df):
    spike_cols = [col for col in ques_ans_cnn_df.columns if 'RPN_Img' in col]
    rpn_train = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='train'][spike_cols].to_numpy()
    rpn_val = ques_ans_cnn_df[ques_ans_cnn_df.dataset=='val'][spike_cols].to_numpy()
    return rpn_train,rpn_val

if __name__ == "__main__":
    try:
        ques_index = load_ques_to_index()
        ans_index = load_answer_to_index()
        preprocessed_data = load_ques_img_mapping()
        ques_ans_df = load_ques_ans_mapping()

        ques_df = generate_ques_embedding(preprocessed_data,ques_ans_df)
        cnn_img_all = get_img_cnn_features()
        rpn_img_all = get_img_rpn_features()

        ques_ans_cnn_df = generate_ques_ans_cnn_df(ques_df,cnn_img_all)
        cnn_train,cnn_val = get_cnn_imgs(ques_ans_cnn_df)
        np.save('cnn_train.npy',cnn_train)
        np.save('cnn_val.npy',cnn_val)

        ques_train,ques_val = get_ques(ques_ans_cnn_df)
        np.save('ques_train.npy',ques_train)
        np.save('ques_val.npy',ques_val)

        ques_ans_rpn_df = generate_ques_ans_rpn_df(ques_df,rpn_img_all)
        rpn_train,rpn_val = get_rpn_imgs(ques_ans_rpn_df)
        np.save('rpn_train.npy',rpn_train)
        np.save('rpn_val.npy',rpn_val)

        ans_encoded_train,ans_encoded_val = get_answer_encoded(ques_ans_cnn_df)
        np.save('ans_encoded_train.npy',ans_encoded_train)
        np.save('ans_encoded_val.npy',ans_encoded_val)

    except:
        del preprocessed_data,ques_ans_df,ques_df,cnn_img_all,rpn_img_all,ques_ans_cnn_df,ques_ans_rpn_df
