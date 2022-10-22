# Airflow 2.x for ML Pipeline 


## Pre-requisites

The whole workshop will work on your local docker environment. 

You should have `docker` and `docker-compose` installed on your machine !

## Getting started


### 1. Clone this repo
```
git clone https://github.com/pycon-ml/airflow_workshop.git
```

### 2. Setup local Airflow environment with Docker

Use docker-compose to build all the required docker images:

```
docker-compose pull
docker-compose build
```

### 3. Use docker-compose to start the applications:

```
docker-compose up
```
### 4. Access services from browser

#### **Airflow**

*UI*: http://localhost:8080

*Username*: airflow

*Password*: airflow

#### **MLflow**

http://localhost:5000

#### **Celery Flower**

http://localhost:5555


## Tear down

Stop and remove containers, networks, images, and volumes

```
docker-compose down
```

## Setup local environment for debug

If there is needs to have local environment to develop and debug, you can use `conda` to create the environment:

```
conda env create -f environment.yml
conda activate airflow_ml
```
