# EC530_Queue_implementation
## Description
This is a Flask API for the training module for the DIYML project. This module allows users to upload their own parameters, start training the data, and get training stats afterwards.

## Required libraries 
- Transformers: this library is used to create a Trainer object to conduct training on the data.
- Datasets
- PyTorch
- Pillow
- Pandas

## Unit tests
- test_upload_parameters: test to see if users can upload their own parameters and see if it is stored in the database as an object.
- test_task_queue: the test first loads the first 100 entries of the food101 dataset from HuggingFace hub and saved it to the local path, then passed that path within the post request to the API to simulate user picking which dataset to run the API on. The image attached in the repository will show the output of the task queue during the test when 2 post requests are sent to the API, which is task 1 will run then task 2.
- test_get_training_stats: test to see if users can retrieve training stats from database

