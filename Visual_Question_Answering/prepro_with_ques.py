def preprocess_question_files():
    train = []
    imdir='COCO_%s_%012d.jpg'
      import json
    
    
    #-----------Load train-val-test data---------------
    with open('v2_OpenEnded_mscoco_train2014_questions.json') as f:
      train_Question = json.load(f)
    
    with open('v2_mscoco_train2014_annotations.json') as f:
      train_Answers = json.load(f)
    
    
    with open('v2_OpenEnded_mscoco_val2014_questions.json') as f:
        val_Question = json.load(f)
    
    with open('v2_mscoco_val2014_annotations.json') as f:
        val_Answers = json.load(f)
        
        
    with open('v2_OpenEnded_mscoco_test2015_questions.json') as f:
        test_ques = json.load(f)
  
    
    subtype = 'train2014'
    for i in range(len(train_Question['questions'])):
     question=train_Question['questions'][i]['question']
     ans=train_Answers['annotations'][i]['multiple_choice_answer']
     question_id=train_Answers['annotations'][i]['question_id']
     image_id=train_Answers['annotations'][i]['image_id']
     image_path=imdir%(subtype, train_Answers['annotations'][i]['image_id'])
     dataset='train'
     
     train.append({'ques_id': question_id, 'img_path': image_path, 'question': question,  'ans': ans,'dataset':dataset})
    
    
    #-----------Preprocess val data---------------
    
    subtype = 'val2014'
    for i in range(len(val_Question['questions'])):
     question=val_Question['questions'][i]['question']
     ans=val_Answers['annotations'][i]['multiple_choice_answer']
     question_id=val_Answers['annotations'][i]['question_id']
     image_id=val_Answers['annotations'][i]['image_id']
     image_path=imdir%(subtype, val_Answers['annotations'][i]['image_id'])
     dataset='val'
    
     train.append({'ques_id': question_id, 'img_path': image_path, 'question': question,  'ans': ans,'dataset':dataset})
    
    json.dump(train, open('vqa_raw_train.json', 'w')) 


if __name__ == "__main__":

    preprocess_question_files()