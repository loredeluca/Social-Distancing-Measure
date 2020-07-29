# Social-Distancing-Measure
This software is a social distancing detector: it use the OpenPose tool to extract a 2D skeleton from the webcam/video and use skeleton joints to measure feet location for each detected person. 
Thanks to the homography it convert feet coordinates in a bird view and compute the distances between each person and its neighbors and generate a warning message when the distance between two persons is less than 2 mt.

The result obtained is the following:

![output](https://github.com/loredeluca/Social-Distancing-Measure/blob/master/output.gif)

### How to install
- Download [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (you can download a simplified version for Mac OS that works on CPU [here](https://github.com/ildoonet/tf-pose-estimation)) to detect people on the scene.
- Make sure you have installed a recent version of OpenCv (version 4.2.0 and later) to generate the bird-view

### How to make it work
You can apply this software to a recorded video or real-time video (like webcam), but a chessboard (or similar) must be present in the scene:
```sh
cap = cv2.VideoCapture(args.video) #recorded video
```
or
```sh
cap = cv2.VideoCapture(args.video0) #real-time video
```
Then, running `socialDistance.py`, if the software does not automatically detect the chessboard, will open a window of the first frame in the video for **calibration** and will be asked to mark 4 points:

- These 4 are used to mark the vertices od the chessboard, which will be used to create the Region of interest (ROI) used to represent the bird-view. The points must be provided in this order: top left, top right, bottom right and bottom left.

Note: make sure to set the correct value in meters of the longest side of the chessboard placed in the scene to the variable 'REFERENCE_DISTANCE'.

After marking these 4 points, with another click anywhere in the scene the social distance detection will begin and the output will show:

- normal video, with the skeleton drawn by openPose and lines indicating the social distancing of people; if the social distance is less than 2 meters, a "Warning message" will start, followed by an acoustic signal.

- bird eye view, with detected people and their safe-distance region.

The details of this work and the functions used are available in the paper ['Image and Video Analysis Project'](INSERIRE LINK GITHUB DELLA RELAZIONE NEL REPOSITORY) in this repository

Some operating examples are available at the following link: https://drive.google.com/drive/folders/1X1h2gBOS2y37HF5bY90luw8SgPwK6XUW

### References
[1] Landing AI, ["AI Tool to Help Customers Monitor Social Distancing in the Workplace"](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/), April 2020
