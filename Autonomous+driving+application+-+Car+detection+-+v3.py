
# coding: utf-8

# # Autonomous driving - Car detection
# 
# Used object detection using the very powerful YOLO model. Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242). 
# 
# **You will learn to**:
# - Use object detection on a car detection dataset
# - Deal with bounding boxes
# - With Non-max suppression
# 

# In[1]:

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

get_ipython().magic('matplotlib inline')


# **Important Note**: As you can see, we import Keras's backend as K. This means that to use a Keras function in this notebook, you will need to write: `K.function(...)`.

# 

# 

# 

# 

# 

# 

# 

# In[2]:

# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = tf.multiply(box_confidence,box_class_probs)
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores,-1)
    box_class_scores = K.max(box_scores,-1)
    ### END CODE HERE ###
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = tf.greater(box_class_scores,threshold)
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    ### END CODE HERE ###
    
    return scores, boxes, classes


# In[3]:

with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            10.7506
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [ 8.42653275  3.27136683 -0.5313437  -4.94137383]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            7
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (?,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (?, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (?,)
#         </td>
#     </tr>
# 
# </table>

# 

# 

# 

# In[57]:

# GRADED FUNCTION: iou

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((xi2-xi1),0)*max((yi2-yi1),0)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1])*(box1[2]- box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2]- box2[0])
    union_area = (box1_area + box2_area) - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###

    return iou


# In[58]:

box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4) 
print("iou = " + str(iou(box1, box2)))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **iou = **
#         </td>
#         <td>
#            0.14285714285714285
#         </td>
#     </tr>
# 
# </table>

# 

# In[6]:

# GRADED FUNCTION: yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold,)
    ### END CODE HERE ###
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    ### END CODE HERE ###
    
    return scores, boxes, classes


# In[7]:

with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            6.9384
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [-5.299932    3.13798141  4.45036697  0.95942086]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            -2.24527
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (10, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
# 
# </table>

# 

# In[8]:

# GRADED FUNCTION: yolo_eval

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE ### 
    
   # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)
    
    ### END CODE HERE ###
    
    return scores, boxes, classes


# In[9]:

with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **scores[2]**
#         </td>
#         <td>
#            138.791
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes[2]**
#         </td>
#         <td>
#            [ 1292.32971191  -278.52166748  3876.98925781  -835.56494141]
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes[2]**
#         </td>
#         <td>
#            54
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **scores.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **boxes.shape**
#         </td>
#         <td>
#            (10, 4)
#         </td>
#     </tr>
# 
#     <tr>
#         <td>
#             **classes.shape**
#         </td>
#         <td>
#            (10,)
#         </td>
#     </tr>
# 
# </table>

# <font color='blue'>
# **Summary for YOLO**:
# - Input image (608, 608, 3)
# - The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
# - After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
#     - Each cell in a 19x19 grid over the input image gives 425 numbers. 
#     - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
#     - 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and and 80 is the number of classes we'd like to detect
# - You then select only few boxes based on:
#     - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
#     - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
# - This gives you YOLO's final output. 

# ## 3 - Test YOLO pretrained model on images

# 

# In[10]:

sess = K.get_session()


# ### 3.1 - Defining classes, anchors and image shape.

# 

# In[11]:

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    


# 

# In[12]:

yolo_model = load_model("model_data/yolo.h5")


# This loads the weights of a trained YOLO model. Here's a summary of the layers your model contains.

# In[13]:

yolo_model.summary()


# **Note**: On some computers, you may see a warning message from Keras. Don't worry about it if you do--it is fine.
# 
# **Reminder**: this model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2).

# 

# In[14]:

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


# You added `yolo_outputs` to your graph. This set of 4 tensors is ready to be used as input by your `yolo_eval` function.

# 

# In[15]:

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


# 

# In[16]:

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = None
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


# Run the following cell on the "test.jpg" image to verify that your function is correct.

# In[17]:

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Found 7 boxes for test.jpg**
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.60 (925, 285) (1045, 374)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.66 (706, 279) (786, 350)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **bus**
#         </td>
#         <td>
#            0.67 (5, 266) (220, 407)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.70 (947, 324) (1280, 705)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.74 (159, 303) (346, 440)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.80 (761, 282) (942, 412)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **car**
#         </td>
#         <td>
#            0.89 (367, 300) (745, 648)
#         </td>
#     </tr>
# </table>

# 

# <font color='blue'>
# **What you should remember**:
# - YOLO is a state-of-the-art object detection model that is fast and accurate
# - It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. 
# - The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
# - You filter through all the boxes using non-max suppression. Specifically: 
#     - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
#     - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
# - Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise. 

# **References**: The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website. 
# - Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
# - Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
# - Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
# - The official YOLO website (https://pjreddie.com/darknet/yolo/) 

# **Car detection dataset**:
# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. We are especially grateful to Brody Huval, Chih Hu and Rahul Patel for collecting and providing this dataset. 
