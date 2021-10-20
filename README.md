# Suicide Detection

## What is the Suicide Detection 
Suicide detection detects whether an input message is suicide related or not. The main focus
in this project is Reddit, so that the database which is used in the model has ocurred with 'suicide' and 'non-suicide'
labelled data taken from Reddit. For the dataset and further read, you may want to visit
https://www.kaggle.com/nikhileswarkomati/suicide-watch.

## How to use it

### Training

Since the dataset which was used have two labels (suicide/non-suicide), we map them as 0 and 1. After stemming, we build
our corpus, using the texts. For vectorizing, we determine the n-gram range and max_features
(which is the same amount of data in our model). After using classifier, we write our model on a file.
First time training the model may take a long time. However, you can use the provided model which was trained with 20000 samples.

### Main

In the main part we call the corpus and the model itself which were written in the training part
After using fit_transform, we build our function to detect if the input is suicide related or not.
You can see below some examples which were taken from online newspaper websites.
```
result:

('Her best interests were not taken into consideration. Sky rang me crying when she found out she was going to Harplands. She should never have been moved there.', 'suicide')
("Coronavirus, or Covid-19, has infected more than 219 million people around the world.The pandemic has had an unprecedented impact on UK life, and for the past year and a half there have been various lockdowns designed to curtail the disease's spread.", 'non-suicide')
('Life and death are equal, there is no difference between survival.', 'suicide')
('New German Media Makers (NdM) is a nonprofit association representing media professionals with immigrant backgrounds. It believes that while Germany has grown more diverse, its media hasn’t. ', 'non-suicide')
```


