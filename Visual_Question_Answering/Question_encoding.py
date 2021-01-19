import sys
import numpy as np
from nltk.tokenize import word_tokenize
import json
import csv
import re

#params = {"input_train_json":"vqa_raw_train.json", "input_test_json": "vqa_raw_test.json" "num_ans": 1000,"output_json":"data_prepro.json"}
params ={"input_train_json": "vqa_raw_train.json",
         #"input_test_json": "vqa_raw_test.json",
         "num_ans": 1000,
         "output_json": "i_to_w_i_to_a.json",
         #"output_h5": "data_prepro.h5",
         "max_length": 26,
         "word_count_threshold": 0,
         "num_test":"0",
         "token_method": "nltk",
         "spacy_data":"spacy_data"}


##FUNCTIONS###


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):
  
    # preprocess all the question
    print ('example processed tokens:')
    for i,img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        elif params['token_method'] == 'spacy':
            txt = [token.norm_ for token in params['spacy'](s)]
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print (txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top words and their counts:')
    print ('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print ('total words:'), total_words
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print ('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print ('inserting the special UNK token')
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top answer and their counts:')
    print ('\n'.join(map(str,cw[:20])))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def encode_question(imgs, params, wtoi):

    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    
    return label_arrays, label_length, question_id


def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi[img['ans']]

    return ans_arrays

def encode_mc_answer(imgs, atoi):
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print ('question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs

def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w,n in count_img.items()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.


    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['img_path'])

    return unique_img, img_pos





def main(params):
    if params['token_method'] == 'spacy':
        print ('loading spaCy tokenizer for NLP')
        params['spacy'] = nlp.English(data_dir=params['spacy_data'])

    imgs_train = json.load(open(params['input_train_json'], 'r'))
   # imgs_test = json.load(open(params['input_test_json'], 'r'))

    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w:i+1 for i,w in enumerate(top_ans)}
    itoa = {i+1:w for i,w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)

    # tokenization question
    imgs_train = prepro_question(imgs_train, params)
    #len(imgs_train)
    #574913
    
    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)
    unique_img_train, img_pos_train = get_unqiue_img(imgs_train)
     # get the answer encoding.
    A = encode_answer(imgs_train, atoi)
 
    # create output h5 file for training set.
    N = len(imgs_train)
    
    
    np.save("ques_train",ques_train)
    np.save("ans_arrays",A)
    np.save("ques_length_train",ques_length_train)
    np.save("question_id_train",question_id_train)
    np.save("img_pos_train",img_pos_train)
    
  
    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
   # out['unique_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    #print( 'wrote ', params['output_json'])

    with open('preprocessed1_whole.csv', 'w', encoding='utf8', newline='') as csvfile:
        fieldnames = ['ques_id', 'img_path', 'question', 'ans', 'processed_tokens', 'final_question','dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(imgs_train)
