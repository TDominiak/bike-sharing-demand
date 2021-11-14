# bike-sharing-demand
## About The Project

This repository contains kaggle `https://www.kaggle.com/c/bike-sharing-demand/` competition solutions. 
Main script return `answer.csv` in appropriate format. File can be found in `data` folder.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* You need python to run the code. I used python 3.8 version. Optionaly: Docker
* You need to download and unpack the [data](https://www.kaggle.com/c/bike-sharing-demand/data) into folder `data` in main directory
* You can create a virtual environment, install libraries and run scripts, or just run the docker image to get the answer


### Run in virtualenv

1. Install packages
   ```sh
   pip install requirements.txt
   ```
2. Run code
   ```sh
   make run
   ```

### Run in docker
1. Run docker
   ```sh
   make docker-run MOUNT=<your-folder-path>/bike-sharing-demand/data 
   ```
