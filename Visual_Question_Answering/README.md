# Visual Question Answering

## Steps for running code
1. Download data from VQA website
2. Run ‘prepro_quest.py’
3. Run ‘Question_encoding.py’
4. Run ‘glove_encoding.py’
5. Run Image processing files to generate CNN vector
6. Make sure you setup Tensorflow object detection API in your local system before executing this code RPN feature extraction code. Follow this link to make your setup: https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/ Also, run the code from the object_detection folder under the research directory.
7. To merge all file, run feature_mapping.py
8. To train final dataset, run vqa_dl_combined_model.py
9. To run pretrained model, load weights.best_full_BN_drop.hdf5 file
10. Attached demo notebook as well. Make sure to run it from object_detection folder under the research directory of Tensorflow object detection API local setup.

## Final Presentation Video Link:
<a name="Visual Question Answering Presentation" href="https://www.youtube.com/watch?v=PDTpBWwNxZs&t">Visual Question Answering Presentation</a>

## Visual Question Answering Demo:
<video width="320" height="240" controls>
	<source src="Demo/VQA_demo.mp4" type="video/mp4">
	Demo/VQA_demo.mp4
</video>

## Requirements (Python3):
* numpy
* pandas
* deepdish
* tensorflow >= 2.0
* keras
* nltk
* six.moves.urllib
* tarfile
* zipfile
* PIL
* matplotlib
* glob
* tkinter
* IPython.display
* Tensorflow Object Detection Faster-RCNN API