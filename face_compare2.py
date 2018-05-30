import cv2
#import cv
import numpy as np
from glob import glob
import os
from sys import argv
from PIL import Image
import math

def path_append(full_path,new_dir):
    return "/".join(full_path.split("/")[:-1])+"/"+new_dir+full_path.split("/")[-1]
        

def normalize_cv(image1,image2,compare_dir):
    #img = Image.open(image1)
    #height < width
    width_heights = []
    width_heights.append(["long_dir/",(100,150)]) 
    #width < height
    width_heights.append(["wide_dir/",(150,100)])
    #equal
    width_heights.append(["equal_dir/",(150,150)])
    #= img.size
    #resize comparison image
    for width_height in width_heights:
        if not os.path.exists(width_height[0]):
            os.mkdir(width_height[0])
        os.chdir(width_height[0])
        img = cv2.imread(image1)
        resized_image1 = cv2.resize(img, width_height[1])
        new_path1 = path_append(image1,width_height[0])
        cv2.imwrite(new_path1,resized_image1)
        img2 = cv2.imread(image2)
        resized_image2 = cv2.resize(img2,width_height[1])
        new_path2 = path_append(image2,width_height[0])
        cv2.imwrite(new_path2,resized_image2)
        os.chdir("../")
        #resize test directory
        for pic in glob(compare_dir+"*"):
            if os.path.isfile(pic):
                full_pic = os.path.abspath(pic)
                im = cv2.imread(full_pic)
                resized_image = cv2.resize(im, width_height[1])
                new_path = path_append(pic,width_height[0])
                cv2.imwrite(new_path,resized_image)


def normalize_pil(image1,image2,compare_dir):
    #img = Image.open(image1)
    #height < width
    width_heights = []
    width_heights.append(["long_dir/",(100,150)]) 
    #width < height
    width_heights.append(["wide_dir/",(150,100)])
    #equal
    width_heights.append(["equal_dir/",(150,150)])
    #= img.size
    #resize comparison image
    for width_height in width_heights:
        if not os.path.exists(width_height[0]):
            os.mkdir(width_height[0])
        os.chdir(width_height[0])
        img = Image.open(image1)
        img.thumbnail(width_height[1],Image.ANTIALIAS)
        new_path1 = path_append(image1,width_height[0])
        img.save(new_path1,"JPEG")
        img2 = Image.open(image2)
        img2.thumbnail(width_height[1],Image.ANTIALIAS)
        new_path2 = path_append(image2,width_height[0])
        img2.save(new_path2,"JPEG")
        os.chdir("../")
        #resize test directory
        for pic in glob(compare_dir+"*"):
            if os.path.isfile(pic):
                full_pic = os.path.abspath(pic)
                im = Image.open(full_pic)
                im.thumbnail(width_height[1],Image.ANTIALIAS)
                new_path = path_append(pic,width_height[0])
                im.save(new_path,"JPEG")


CONFIDENCE_THRESHOLD = 100.0
ave_confidence = 0
num_recognizers = 3
recog = {}
recog["eigen"] = cv2.face.EigenFaceRecognizer_create()
recog["fisher"] = cv2.face.FisherFaceRecognizer_create()
recog["lbph"] = cv2.face.LBPHFaceRecognizer_create()

#load the data initial file
filename = os.path.abspath(argv[1])
compare = os.path.abspath(argv[2])
#normalize other faces
base_picture_dir = argv[3] + "/"
normalize_cv(filename,compare,argv[3])

dirs = ["wide_dir/","long_dir/","equal_dir/"]
dirs = [base_picture_dir+elem for elem in dirs]
#generate test data

for Dir in dirs:
    new_filename = path_append(filename,Dir)
    face = cv2.imread(new_filename,0)
    try:
        face,label = face[:, :], 1
    except:
        import code
        code.interact(local=locals())
    #load comparison face
    new_compare = path_append(compare,Dir)
    compare_face = cv2.imread(new_compare, 0)
    compare_face, compare_label = compare_face[:,:], 2

    images,labels = [],[]
    images.append(np.asarray(face))
    images.append(np.asarray(compare_face))
    labels.append(label)
    labels.append(compare_label)

    image_array = np.asarray(images)
    label_array = np.asarray(labels)
    for recognizer in recog.keys():
        recog[recognizer].train(image_array,label_array)

    print(Dir+"\n\n")
    test_images = glob(Dir+"*")
    test_images = [(np.asarray(cv2.imread(img,0)[:,:]),img) for img in test_images]
    possible_matches = []
    for t_face,name in test_images:
        t_labels = []
        for recognizer in recog.keys():
            try:
                [label, confidence] = recog[recognizer].predict(t_face)
                possible_matches.append({"name":name,"confidence":confidence,"recognizer":recognizer})
            except:
                continue
    minimum = 400
    epsilon = 1
    close_enough = []
    average = 0
    for m in possible_matches:
        if m["recognizer"] == "lbph":
            average += m["confidence"]
            if m["confidence"] < minimum:
                minimum = m["confidence"]
    minimum += epsilon
    try:
        average /= float(len(possible_matches))
    except:
        import code
        code.interact(local=locals())
    Median = np.median([m["confidence"] for m in possible_matches])
    for m in possible_matches:
        if m["recognizer"] == "lbph":
            if m["confidence"] == minimum or m["confidence"] < (minimum + average):
                close_enough.append(m)
            #if m["confidence"] < Median:
            #    close_enough.append(m)
    for i in close_enough:
        print(i)
