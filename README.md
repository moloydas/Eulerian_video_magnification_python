# Eulerian_video_magnification_python
This repository is a python implementation of Eulerian Video Magnification that can reveal tiny motion and can make the color changes due to blood flow apparent. 

# Run
python euler_mag.py input_video.avi

# Libraries
- Scipy
- Opencv
- Numpy

# Demo
## Source Video

https://user-images.githubusercontent.com/20353960/141691709-699642fc-b68f-45a3-a9d8-ee1a43655253.mp4

## Color Magnified Video

https://user-images.githubusercontent.com/20353960/141691797-100ade3b-9f4b-4cca-a019-63188c41c7b7.mp4

## Source Video

https://user-images.githubusercontent.com/20353960/141691860-ac3a4c22-fe09-422c-8423-dee8ee548abc.mp4

## Motion Magnified Video

https://user-images.githubusercontent.com/20353960/141691846-d5334d87-bd41-4a20-883c-8fedbfc8a1ab.mp4

## Source Video
## My face video

https://user-images.githubusercontent.com/20353960/141692550-c3018b2a-4055-4585-af18-583edc83aea3.mp4

## My face video with magnified motion and color

https://user-images.githubusercontent.com/20353960/141692571-4d051b3a-ec00-4d7c-b1ea-6cf1b76e6e16.mp4

A temporal view of my cheek in the above video. Here, we can easily see the periodic color change (between red and green) with time (horizontal axis) indicating the flow of blood.
![sliced](https://user-images.githubusercontent.com/20353960/141746680-1276f38e-e4a9-4301-afe2-3d48fe9eca3a.png)

# References
@article{Wu12Eulerian,
  author = {Hao-Yu Wu and Michael Rubinstein and Eugene Shih and John Guttag and Fr\'{e}do Durand and
  William T. Freeman},
  title = {Eulerian Video Magnification for Revealing Subtle Changes in the World},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH 2012)},
  year = {2012},
  volume = {31},
  number = {4},
}

# Limitations
- Color magnification is dependent on skin tone. Dark regions of skin shows very little changes
- Motion magniication makes the frame noisy as well and becomes a problem in case of large background motions. 
