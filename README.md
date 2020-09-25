# Hyper: Stellar Classifier
Applied Semester Project Group 42

Members:
* Kane Bruce
* Eran Rozewski
* Jessica Berry

Project Name: Stellar Classifier

Problem Statement: Telescopes and satellites obtain huge amounts of data every moment and that data needs to be processed. For decades, people did the processing and classification of images and information by hand. Over time, there have been several competitions on developing machine learning algorithms that can perform the data processing in the shortest time with the least amount of error. 

Proposed Solution: Find the best-performing machine learning model among a variety of image classification models that yields the lowest RMSE score upon training and validation, using TensorFlow and Keras. The model to tune is the convolutional neural network, considered by most professionals to be one of the most accurate image classification models to date. Hyperparameters to tune include layer size and layout, activation functions, kernel regularizer, and optimization and loss functions. 

Data: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

## How to run
* Create a new branch, if you plan to play around with parameters or building new models, etc!
  * `>>> Never commit to master directly!! <<<`
  
* Open `__main__.py`, to see the list of packages you may need to install at the top.
  * You may have to run it a couple of times to figure out what packages you're missing... sorry!
  * Feel free to get a feel for the setup of the files as well! Models go in the `models` directory. I used a lot of python tricks to make it really flexible. -Kane
  
* Make sure you set up TensorFlow with GPU support, if possible. Try to run on the fastest storage you have, too!
  * A single test/train/validation run of the `models.cnn.CNN` model takes ~1.5 hours on an Nvidia RTX 2070, Samsung 970 Evo NVMe SSD
  
* Acquire the dataset zip files, and create a `data` folder in the root directory. Be sure to unzip into this directory:
  * folder: `images_training_rev1`
  * folder: `images_test_rev1`
  * file: `training_solutions_rev1.csv`
  
* When you feel ready to run it after messing with parameters, etc: `python __main__.py`
  * It will take some time. Get a cup of tea and relax!
  * The script will create an `out` directory for you at the end. Statistics will appear in the terminal as well, such as epoch training progress and at the end, RMSE score.
