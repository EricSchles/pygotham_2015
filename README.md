#BUILDING TOOLS FOR SOCIAL GOOD

_By_ 

_Eric_ _Schles_

##Background/About:

Academic: Comp Sci, Math, Econ
What I do:

* [wikitongues](http://www.wikitongues.org/)

* [heatseeknyc](http://heatseeknyc.com/)

* demand abolition - No link, we aren't proud of the website

How you can help:



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

The `face_compare2.py` contains how I do face comparison with open CV.  It works quiet well, but I do confess it is not my most well organized piece of code.  I have not yet learned how to be elegant with image manipulation, alas, the struggles of perfection in a sea of desires and things one aspires to be perfect at.  But such is the calamity of curious minds, and it is something I am proud of, to be pulled in so many possible directions.

In any event, there is a lot to unpack in that file and so I'm simply going to explain some high level steps:

First we instantiate our recognizers:
```
recog = {}
recog["eigen"] = cv2.face.createEigenFaceRecognizer()
recog["fisher"] = cv2.face.createFisherFaceRecognizer()
recog["lbph"] = cv2.face.createLBPHFaceRecognizer()
```

Notice that recently OpenCV has changed and now the face recognizers live in a seperate supplemental library.  The core of OpenCV can be found here: https://github.com/Itseez/opencv and the face recognizers can be found here: https://github.com/Itseez/opencv_contrib.  I'm not sure why they decided to move such a handy set of tools out into some contrib repo, but that is how it is.

The next thing to do is normalize the pictures:

normalize_cv(filename,compare,directory_of_pictures)

In this case, this means making them all the same height and width, because the face comparison algorithms require this.  I'm not sure why this is the case.  

Next we read in a picture and hand label it 1:

```
face = cv2.imread(new_filename,0)
face,label = face[:, :], 1
```

And we read in a different picture with a different face and hand label it 2:

```
compare_face = cv2.imread(new_compare, 0)
compare_face, compare_label = compare_face[:,:], 2
```

This establishes the base comparison for what will be considered a face we are searching for and a face that is different from the one we want.  Perhaps using 2 isn't the best, because the distance isn't great enough, but this is intended to be a toy example, you should feel free to tune these "magic" numbers to improve the precision of your own code.  

Next we train the each recognizer on the training data:

```
for recognizer in recog.keys():
    recog[recognizer].train(image_array,label_array)
```

And then we compare all the trained data against the directory of pictures:

```
test_images = [(np.asarray(cv2.imread(img,0)[:,:]),img) for img in test_images]
possible_matches = []
for t_face,name in test_images:
    t_labels = []
    for recognizer in recog.keys():
        try:
            [label, confidence] = recog[recognizer].predict(t_face)
            possible_matches.append({"name":name,"confidence":confidence,"recognizer":recognizer})
```

Notice that we simply need to call the predict function on each image in our directory.

Finally, we simply can print out all the possible matches:

```
for i in possible_matches:
	print i
```

Running this code is fairly simple and can be accomplished with the following:

`python face_compare2.py [first picture] [baseline comparison picture] [directory of pictures to search against]`

Note: For this code to work from this repo, you'll need to install OpenCV from the github's I link to above.

##Building/Using a GIS system, for free

There are a lot of solutions out there for python and GIS.  GeoDjango is the most packaged, with the least amount of extra installs, that I know of.  

First we'll install Geodjango, which luckily comes with Django1.8.  So I'll you'll need to do really is install django, instructions on how to do that can be found [here](https://docs.djangoproject.com/en/1.8/topics/install/), assuming you want the source or anything else.

But really all you need to do is: `sudo pip install django`

Before we can get started with building we'll need to install PostGIS, steps on how to do that can be found [here](http://www.saintsjd.com/2014/08/13/howto-install-postgis-on-ubuntu-trusty.html)

Next we'll need to install the postgres python module - pyscopg2: `sudo pip install pyscopg2`

Next all we need to do is set up GeoDjango to work with PostGIS, and we are off to the races.

Unfortunately the GeoDjango documentation for the first example is confusion and leaves out a few steps, so first we'll have to correct that:

Please note these directions are for Ubuntu 14.04

Installation:

Python:
`sudo pip install django #make sure you are installing django 1.8` 

`sudo pip install pyscopg2`

PostGres:

`sudo apt-get update`

`sudo apt-get install -y postgresql postgresql-contrib`

Testing postgres install:

`sudo -u postgres createuser -P USER_NAME_HERE`

`sudo -u postgres createdb -O USER_NAME_HERE DATABASE_NAME_HERE`

`psql -h localhost -U USER_NAME_HERE DATABASE_NAME_HERE`

Adding PostGIS support:

`sudo apt-get install -y postgis postgresql-9.3-postgis-2.1`

`sudo -u postgres psql -c "CREATE EXTENSION postgis; CREATE EXTENSION postgis_topology;" DATABASE_NAME_HERE`

changing everything to trusted, rather than requiring authentication - DO THIS FOR LOCAL DEVELOPMENT ONLY!!!

`sudo emacs /etc/postgresql/9.1/main/pg_hba.conf`

Change line:

`local   all             postgres                                peer`

To

`local   all             postgres                                trust`

Then restart postgres:

`sudo service postgresql restart`

Getting started with geodjango:

Now we are ready to get started:

`django-admin startproject geodjango`

`cd geodjango`

`python manage.py startapp world`

Now we'll go into the settings.py file:

`emacs geodjango/settings.py`

and edit the databases connection to look like this:

```
DATABASES = {
    'default': {
         'ENGINE': 'django.contrib.gis.db.backends.postgis',
         'NAME': 'geodjango',
         'USER': 'geo',
     }
} 
```

Notice that we haven't created the 'geodjango'-database so we'll do that now:

`sudo -u postgres createuser -P geo`

`sudo -u postgres createdb -O geo geodjango`

`sudo -u postgres psql -c "CREATE EXTENSION postgis; CREATE EXTENSION postgis_topology;" geodjango`

we'll also need to edit the installed aps, in the same file:

```
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',
    'world'
)
```

Great, now we can save and close that.

Next we'll need some data to visualize:

`mkdir world/data`

`cd world/data`

`wget http://thematicmapping.org/downloads/TM_WORLD_BORDERS-0.3.zip`

`unzip TM_WORLD_BORDERS-0.3.zip`

`cd ../..`

Now let's inspect our data so we now how our model should look - we should try to be consistent with how the data is annotated for portability and extensibility.  

For this we'll need gdal - `sudo apt-get install libgdal-dev python-gdal gdal-bin # the python library is unnecessary but nice to have :)`

Now we can inspect the annotation in the shapefile of our geospatial data:

`ogrinfo -so world/data/TM_WORLD_BORDERS-0.3.shp TM_WORLD_BORDERS-0.3`

We'll use this output to map to our models.py file:

`emacs world/models.py` and type:

```
from django.contrib.gis.db import models

class WorldBorder(models.Model):
    # Regular Django fields corresponding to the attributes in the
    # world borders shapefile.
    name = models.CharField(max_length=50)
    area = models.IntegerField()
    pop2005 = models.IntegerField('Population 2005')
    fips = models.CharField('FIPS Code', max_length=2)
    iso2 = models.CharField('2 Digit ISO', max_length=2)
    iso3 = models.CharField('3 Digit ISO', max_length=3)
    un = models.IntegerField('United Nations Code')
    region = models.IntegerField('Region Code')
    subregion = models.IntegerField('Sub-Region Code')
    lon = models.FloatField()
    lat = models.FloatField()

    # GeoDjango-specific: a geometry field (MultiPolygonField), and
    # overriding the default manager with a GeoManager instance.
    mpoly = models.MultiPolygonField()
    objects = models.GeoManager()

    # Returns the string representation of the model.
    def __str__(self):              # __unicode__ on Python 2
        return self.name
```

We are now ready to run our first migration :)

`python manage.py makemigrations`


`python manage.py sqlmigreate world 0001`


`python manage.py migrate`

Making our first map:

Now that we've done all the setup, we can leverage geodjango immmediately to create an interactive map!

We'll need to make a few small edits to some files, but essetentially, we are done:

open the admin.py file and type the following:

emacs world/admin.py:

```
from django.contrib.gis import admin
from models import WorldBorder

admin.site.register(WorldBorder, admin.GeoModelAdmin)
```

Finally we simply need create our admin credentials:

`python manage.py createsuperuser`
`python manage.py runserver`

Now head over to [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/)

and enter your credentials.  Now you have a GIS system that you can play with, load data into, and get analysis out of!

I found the setup for GeoDjango to be slightly outdated and confusing and so I thought it was important to spend time on it instead of showing you examples of how to use GeoDjango, for examples of this I recommend:

GeoDjango Specific tutorials and examples:

Basic:

http://blog.mathieu-leplatre.info/geodjango-maps-with-leaflet.html

http://invisibleroads.com/tutorials/geodjango-googlemaps-build.html

http://davidwilson.me/2013/09/30/Colorado-Geology-GeoDjango-Tutorial/


Advanced:

http://blog.apps.chicagotribune.com/category/data-visualization/

Visualizing Crime data with GIS:

http://flowingdata.com/2009/06/23/20-visualizations-to-understand-crime/

Where to get your own data:

https://data.cityofchicago.org/

https://nycopendata.socrata.com/

How to do Geo-Python things with other tools:

https://2015.foss4g-na.org/sites/default/files/slides/Installation%20Guide_%20Spatial%20Data%20Analysis%20in%20Python.pdf


Questions?

contact: 

gmail: ericschles@gmail.com

twitter: @EricSchles
