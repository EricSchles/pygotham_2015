#BUILDING TOOLS FOR SOCIAL GOOD

_By_ 

_Eric_ _Schles_

##Outline:
Computational tools are becoming increasingly necessary in the world of social good. There is an increasing need for data science, automation, search and distributed computation. 

In this talk I will take listeners through:

 * building a image search database with python and a few libraries, 
 * how to build geographic information systems with geodjango, 
 * And how to process and extract information from pdfs.

##The Generalities

The real point of this talk is to understand that using the standards of technological innovation from industry - search, visualization, and data processing; we can make government and non-profits innovative, fast paced, precise, and powerful.

Right now the problem is, technology has not been applied to solving social problems.  This is in part because the private sector just pays better, and in part because government agencies haven't full internalized what having a team of developers can do for them.  And so we can apply the techniques of the last 30 or so years, instantaneously, allowing for monumental gains in productivity.

The place most ripe for innovation: _The_ _local_ _government_

The goal of this talk is to provide three suggestions for how you might innovate a local government agency, using free open source software, and beatifully simple python.  Most government shops are C, C++, or Java.  But they are using main frame style paradigms.  In fact, we basically used a mainframe set up at the Manhattan DA.  (I know I shutter whenever I think about it too)

So clearly there is room for growth.  Other things I won't mention in this talk include:  

* APIs, Restful services
* Cluster Computing
* Advanced machine learning for classication and prediction
* Web Scraping

But you can check out all of these topics in my [forth coming book](https://github.com/EricSchles/book_tools) from Orielly Media :)

##Extracting Information from PDFs

###Installation

Pandas - `sudo pip install pandas` or `sudo apt-get install python-pandas` #ubuntu only
Poppler-utils:
 * On ubuntu - http://poppler.freedesktop.org/
 * On Windows - http://blog.alivate.com.au/poppler-windows/

Note for windows you'll need to set the environment variables or pdftotext.  For information on how to do this please see [this guide](http://www.itechtalk.com/thread3595.html)

The following piece of code can be found in the chapter1 section of the github:

```
from subprocess import call
from sys import argv
import pandas as pd
def transform(filename):
    call(["pdftotext","-layout",filename])
    return filename.split(".")[0] + ".txt"

def segment(contents):
    relevant = []
    start = False
    for line in contents:
        if "PROSECUTIONS CONVICTIONS" in line:
            start = True
        if "The above statistics are estimates only, given the lack" in line:
            start = False
        if start:
            relevant.append(line)
    return relevant

def parse(relevant):
    tmp = {}
    df = pd.DataFrame()
    for line in relevant:
        split_up = line.split(" ")
        # a row - 2008 5,212 (312) 2,983 (104) 30,961 26
        split_up = [elem for elem in split_up if elem != '']
    
        if len(split_up) == 7:
            tmp["year"] = split_up[0]
            tmp["prosecutions"] =split_up[1]
            tmp["convictions"] = split_up[3]
            tmp["victims identified"] = split_up[5]
            tmp["new or ammended legistaltion"] = split_up[6]
            #print tmp
            df = df.append(tmp,ignore_index=True)
    return df

if __name__ == '__main__':
    txt_file = transform(argv[1])
    text = open(txt_file,"r").read().decode("ascii","ignore")
    contents = text.split("\n")
    relevant = segment(contents)
    df = parse(relevant)
    df.to_csv("results.csv")
```

This code parses the pdf - trafficking_report.pdf which can also be found in the github.  This is a very long report, but what we care about is one table in particular (by way of example).  This table can be found on page 45 of the report and it details arrest details about human traffickers internationally.  

Let's walk through each of the methods, which are defined in the order they are called.

```
def transform(filename):
    call(["pdftotext","-layout",filename])
    return filename.split(".")[0] + ".txt"
```

The transform method calls pdftotext, a command line utility that transforms a pdf into a .txt document.  Notice the use of the layout flag which preserves the formatting as much as possible from our pdf file, this is crucial to ensuring our .txt document will be easily parsable.

```
def segment(contents):
    relevant = []
    start = False
    for line in contents:
        if "PROSECUTIONS CONVICTIONS" in line:
            start = True
        if "The above statistics are estimates only, given the lack" in line:
            start = False
        if start:
            relevant.append(line)
    return relevant
```

In the segmentation step we find the text invariants that start and end the section of the document we wish to parse.  Here the start  invariant is `"PROSECUTIONS CONVICTIONS"` and the end invariant is `"The above statistics are estimates only, given the lack"`.  What's returned is a simple dictionary of the relevant lines of text from the transformed .txt document.

```
def parse(relevant):
    tmp = {}
    df = pd.DataFrame()
    for line in relevant:
        split_up = line.split(" ")
        # a row - 2008 5,212 (312) 2,983 (104) 30,961 26
        split_up = [elem for elem in split_up if elem != '']
    
        if len(split_up) == 7:
            tmp["year"] = split_up[0]
            tmp["prosecutions"] =split_up[1]
            tmp["convictions"] = split_up[3]
            tmp["victims identified"] = split_up[5]
            tmp["new or ammended legistaltion"] = split_up[6]
            #print tmp
            df = df.append(tmp,ignore_index=True)
    return df
```

The final section is the parse method.  Here we create a dictionary which will be appended to the pandas DataFrame that we'll make use of.  The reason for storing things in a pandas dataframe is because of the ease of transformation to other persistent file stores such as CSV, EXCEL, a database connection, and others.  Notice that inside the loop we split up the line by whitespace, since this is a table, we should expect tabular data to appear in the same position on different lines.  Also notice that we expect the size of each split up line to be of length 7.  Not that this does not capture a few of the lines in the original pdf, it is left as an exercise to handle these minor cases.  Length is not the best metric for ensuring you are processing the correct information, but the intention of this code is to be as readible as possible, not necessarily as sophisticated as possible.  Other ways you can check to ensure your scraping the correct text is with regex, checking for certain character invariants that should appear on every line, or by using advanced machine learning techniques which we will talk about in the next section.  Finally, the tmp dictionary is assigned each of the values from the table and appended to the dataframe.  Notice that we only need to create tmp dictionary once and then simply overwrite it's contents on each loop through the relevant content.  Then we simply return the dataframe to main.  Here a single line is used to send the dataframe to a csv: 

`df.to_csv("results.csv")` 

And we are done!

There are other, more elegant ways of parsing pdfs, like those found in [this chapter](https://automatetheboringstuff.com/chapter13/) of automate the boring stuff.  Unfortunately, using a library like PyPDF2 won't work in all cases.  So while my method is certainly not elegant, it is robust and will work 99% of the time.  


##Search for images

Another important innovation has been search.  We've more or less figured search out, and for the rest of the world, with databases, this is very useful, but for most folks working in government, things are still stored in a file system.  If you want to do file search efficiently, just use [Solr](http://lucene.apache.org/solr/).

However, after text based documents, the second biggest need for search is images.  There are some standard tools out there, but it's actually worth it to play around with your own solution.  Sometimes you'll need some custom stuff, sometimes you'll need to something special, like my boss asked me to do - image recognition and search off of one picture.  I had to get creative since usually results are not that good for a single picture against other pictures, which are not the same shot.  To be clear my task was this:  

If someone was in a picture, find them in all other pictures they appear in. 

No easy feat.  But I got pretty close.  I ended up settling on two techniques/tools - OpenCVs face comparison algorithms and a wonderful post by the writer of [pyimagesearch](http://www.pyimagesearch.com/).  If you haven't seen it, I recommend checking out that blog, it's full of great posts about computer vision.  

So face comparison searches and compares faces against each other, producing a distance metric, the smaller the distance between two faces, IE the smaller the output, the more likely they are the same person's face.  For pyimagesearch, I made use of their [backgroud comparison post](http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/).  In this post Adrian lays out how to compare the backgrounds of two images - by transforming the pictures into histograms, using a visual bag of words, and then making use of a chi^2 distance, to build an index.  Finally, we search across the index to look for a picture or a similar picture in our set of pictures.  From this we can see if the face and background match for a particular picture - if they do, it's probably the same person.  But each of these metrics own their own is powerful for investigations.  Then we can see who else has been in a the same room, and thus who else knows each other.  We can also see this if we can find multiple people in the same picture.  

One note - we could have used Caffe for the background image search.  The trouble with Caffe is, I find it extremely difficult to install.  Also, I'm not terribly good at C++, which is a major drawback.  The following code is written entirely in python (or wrappers in python).  Caffe does have python wrappers for some of its stuff, and it is a wonderful library if you have the computational power to truly leverage it, but most of the time in government you don't have the ability to install whatever you want and you definitely don't have the computational power to leverage such a tool.

Background compare:


`index_search.py`:

```
import numpy as np
import cv2
import argparse
import csv
from glob import glob


args = argparse.ArgumentParser()
args.add_argument("-d","--dataset",help="Path to the directory that contains the images to be indexed")
args.add_argument("-i","--index",help="Path to where the computed index will be stored")
args.add_argument("-q","--query",help="Path to the query image")
args.add_argument("-r","--result-path",help="Path to the result path")

args = vars(args.parse_args())

class ColorDescriptor:
    def __init__(self,bins):
        self.bins = bins

    #off_center means that we expect no people in the center of the picture
    #this is the feature construction and processing function
    #cX,cY stand for center X and center Y, aka the middle of the picture
    #off_center creates four quadrants for the picture
    #full_picture checks the full picture and only creates one histogram per picture
    #face_compare creates a segmentation around possible faces, and uses only the face to create a histogram
    def describe(self,image,off_center=False,full_picture=False,face_compare=False):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        features = []
        (h,w) = image.shape[:2]
        (cX,cY) = (int(w*0.5), int(h*0.5))
        segments = [(0,cX,0,cY),(cX,w,0,cY),(cX,w,cY,h),(0,cX,cY,h)]
        if off_center:
            for (startX,endX,startY,endY) in segments:
                cornerMask = np.zeros(image.shape[:2],dtype="uint8")
                cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
                hist = self.histogram(image,cornerMask)
                features.extend(hist)
        elif full_picture:
            hist = self.histogram(image,None)
            features.extend(hist)
        elif face_compare:
            cascPath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascPath)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(25, 25),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            for (x,y,w,h) in faces:
                cornerMask = np.zeros(image.shape[:2],dtype="uint8") 
                cv2.rectangle(cornerMask, (x,y), (x+w, y+h), 255, -1)
                hist = self.histogram(image,cornerMask)
                features.extend(hist)
        else:
            ellipMask = np.zeros(image.shape[:2],dtype="uint8")
            (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
            cv2.ellipse(ellipMask, (cX, cY), (axesX,axesY), 0,0,360,255,-1)
            for (startX,endX,startY,endY) in segments:
                cornerMask = np.zeros(image.shape[:2],dtype="uint8")
                cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
                cornerMask = cv2.subtract(cornerMask,ellipMask)
                hist = self.histogram(image,cornerMask)
                features.extend(hist)
        return features
    
    def histogram(self,image,mask):
        hist = cv2.calcHist([image],[0,1,2],mask,self.bins,
        [0, 180, 0, 256, 0, 256])
        hist2 = cv2.normalize(hist,np.zeros(image.shape[:2],dtype="uint8")).flatten()
        return hist2

class Searcher:
    def __init__ (self,indexPath):
        self.indexPath = indexPath
    def search(self, queryFeatures, limit=10):
        results = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            try:
                for row in reader:
                    features = [float(x) for x in row[1:] if x!= '']
                    d = self.chi2_distance(features, queryFeatures)
                    results[row[0]] = d
            except IndexError:
                pass
        results = sorted([(v,k) for (k,v) in results.items()])
        return results[:limit]
    def chi2_distance(self,histA,histB, eps = 1e-10):
        d = 0.5 * np.sum([((a-b)**2) / (a+b+eps)
                          for (a,b) in zip(histA,histB)])
        return d

if __name__ == "__main__":
    #i have no idea where these numbers came from.. check - http://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
    cd = ColorDescriptor((8,12,3))

    if args["index"] and args["dataset"]:
        with open(args["index"],"w") as output:
            for img_type in [".png",".jpg",".PNG",".JPG",".img",".jpeg",".jif",".jfif",".jp2",".tif",".tiff"]:
                for imagePath in glob(args["dataset"] + "/*" + img_type):
                    imageID = imagePath[imagePath.rfind("/") + 1:]
                    image = cv2.imread(imagePath)
                    features = cd.describe(image)
                    features = [str(f) for f in features]
                    output.write("%s,%s\n" % (imageID, ",".join(features)))
    if args["index"] and args["query"]:
        query = cv2.imread(args["query"])
        features = cd.describe(query)
        searcher = Searcher(args["index"])
        results = searcher.search(features,limit=4)
        
        cv2.imshow("query",query)
        for (score,resultID) in results:
            result = cv2.imread(args["result_path"] + "/" + resultID)
            cv2.imshow("Result",result)
            cv2.waitKey(0)
```         

Rather than explaining all this code, because Adrian explains all of it and extremely well, I'll simply explain how to use it:

`python index_search.py -d [directory of pictures] -i [name of file] #create an index`

`python index_search.py -i [name of index file] -q [path to query picture] -r [path to picture directory] #search for a picture`

A specific example: 
    
`python index_search.py -d pic_db/ -i index.csv` 

`python index_search.py -i index.csv -q person.jpg -r pic_db/`


So here's how the code works:

First let's look at the main method:

```
cd = ColorDescriptor((8,12,3))

if args["index"] and args["dataset"]:
    with open(args["index"],"w") as output:
        for img_type in [".png",".jpg",".PNG",".JPG",".img",".jpeg",".jif",".jfif",".jp2",".tif",".tiff"]:
            for imagePath in glob(args["dataset"] + "/*" + img_type):
                imageID = imagePath[imagePath.rfind("/") + 1:]
                image = cv2.imread(imagePath)
                features = cd.describe(image)
                features = [str(f) for f in features]
                output.write("%s,%s\n" % (imageID, ",".join(features)))
if args["index"] and args["query"]:
    query = cv2.imread(args["query"])
    features = cd.describe(query)
    searcher = Searcher(args["index"])
    results = searcher.search(features,limit=4)
    
    cv2.imshow("query",query)
    for (score,resultID) in results:
        result = cv2.imread(args["result_path"] + "/" + resultID)
        cv2.imshow("Result",result)
        cv2.waitKey(0)
```

As you can see there isn't a ton going on here, we loop through all the images in the pictures folder, doing the feature transformations on each picture and then generating an index with all the feature-histogram-vectors.  Now let's dig a little deeper into the describe method.  The describe method acts on the matrix representation of an image.  The image is a matrix at this point, because it was read in with open cv's imread method.  So really the describe method is a matrix transformation, which produces features according to some rules about the image.  The way in which you decide to describe images, turns out to be very very important.  Essentially the describe is doing a segmentation and then a transformation.

One fun thing I did was, I allowed you to segment pictures by face, into four corners, or the way Adrian does it; with four corners and an elliptical in the center.  My ways don't always yield good results, and in fact more than half the time Adrians method works the best, but sometimes, my methods have been effective so I will mention them here.

To understand that we'll look at the segmentation methods one at a time:

```
for (startX,endX,startY,endY) in segments:
    cornerMask = np.zeros(image.shape[:2],dtype="uint8")
    cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
    hist = self.histogram(image,cornerMask)
    features.extend(hist)
```

Here we setup a cornerMask which is a matrix of zeroes.  Notice that we apply the rectangle method, which takes in starting and finishing coordinates and then 'draws' a rectangle on the cornerMask.  Then we create a histogram with the segmentation of the image defined by the rectangle we 'drew' which acts as sort of a boundary.  Depending on how you draw your segmentation, will effect your histogram and thus your features.  

The full picture is uninteresting, because it is just the histogram transformation with no segmentation.  

The next interesting segmentation is one done with semantic meaning built in - finding the startX,startY and endX,endY from the faces in the picture:

```
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(25, 25),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

for (x,y,w,h) in faces:
    cornerMask = np.zeros(image.shape[:2],dtype="uint8") 
    cv2.rectangle(cornerMask, (x,y), (x+w, y+h), 255, -1)
    hist = self.histogram(image,cornerMask)
    features.extend(hist)
```

Here we use opencv's face detection classifier to find all the 'boxes' around the potential faces in the image.  This is then used to create our rectangles, simiarly to the previous example.  However, now rather than segmenting on corners, we segment on faces.  We then use this segmentation to create histograms and finally feature vectors of just the faces.  Unfortunately, in practice this technique tends to perform the worst, for now.  I need to figure out how to better tune my parameters so that this is reliable, or I may abandon it as an idea.  But still, it was fun to try!

The final method is what Adrian did:

```
ellipMask = np.zeros(image.shape[:2],dtype="uint8")
(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
cv2.ellipse(ellipMask, (cX, cY), (axesX,axesY), 0,0,360,255,-1)
for (startX,endX,startY,endY) in segments:
    cornerMask = np.zeros(image.shape[:2],dtype="uint8")
    cv2.rectangle(cornerMask,(startX,startY),(endX,endY),255,-1)
    cornerMask = cv2.subtract(cornerMask,ellipMask)
    hist = self.histogram(image,cornerMask)
    features.extend(hist)
```

Creating an elliptical mask for the center of the picture and generating segments for the remaining four corners and then subtracting the center from the four corners.  Nothing else is new here and thus it can largely be ignored.

The next important piece of the main function is performing a query which is captured here:

```
query = cv2.imread(args["query"])
features = cd.describe(query)
searcher = Searcher(args["index"])
results = searcher.search(features,limit=4)
```

Notice that we describe the query image the same way we described the index.  This is imperative - if we describe our feature vectors differently in indexing and querying, our queries won't work.  

The search function is fairly simple:

```
results = {}
with open(self.indexPath) as f:
    reader = csv.reader(f)
    try:
        for row in reader:
            features = [float(x) for x in row[1:] if x!= '']
            d = self.chi2_distance(features, queryFeatures)
            results[row[0]] = d
    except IndexError:
        pass
results = sorted([(v,k) for (k,v) in results.items()])
return results[:limit]
```

Essentially, it applies the chi^2 distance function to each of the features in the index against the query image's features.  The results are returned in sorted order, lowest to highest.  This is because if the distance is smallest, they are the images can be expected to be the least different.  


###Understanding Face Comparison, With OpenCV



