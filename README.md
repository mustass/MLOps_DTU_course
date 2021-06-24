transformers-group-octupus
==============================

This project is made for the course Machine Learning Operations at the Technical University of Denmark. It has been made in collaboration of the students and acts as a exam project. The focus has been to make a machine learning model operationable, i.e. being able to access a trained model for the problem considered remotely while monitoring the performance to ensure the model functions as expected. The problem considered is text classification and specifically the model should classify what a series of amazon reviews are reviewing. To solve this, a BERT model based on HuggingFace will be employed. The project has been structured using the cookiecutter approach which means the project is organised in the following manner. To run the code it is necessary to clone the repository and run the main with the config file.



## Installation and running when developing

First we create an environment and it is ABSOLUTELY NECESSARY to have Python version 3.7 for Azure to not be a pain in the ass. 

```bash
conda create --name [your_environment_name] python=3.7
```

Then we run:

```bash
conda activate [your_environment_name]
```

After that: 

```bash
pip install -r requirements.txt
```

And lastly to have the correct version of the internal package `src` we do: 

```bash
pip install -e .
```
or 
```bash
conda develop .
```
Then things should work :) 

FOR CLOUD we do need to keep 
```python
-e git+https://github.com/stas97/transformers-group-octupus.git@master#egg=src
```
in `requirements.txt`
This will install the `src` package from GitHub.

PLEASE DO NOT DO 
```bash
pip freeze
```
If you need a package, add it manually and without a specified version. And if you have added a package, it's nice if you could test that recreating the environment with 
```bash
pip install -r requirements.txt
```
will work in a fresh environment. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Config files for experiments, test and deploying
    ├── .giuthub/workflows <- GitHub workflows for test and other fun
    ├── data (local use)
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Sources and materials used during development
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │   └── fetch_dataset.py
    │   │   └── lightning_data_module.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py <- training the model 
    │   │   └── hp_tuning.py   <- Optuna
    │   │   └── model.py       <- Model definition
    │   │   └── main.py        <- Main file to run when training and/or deploying
    │   │   └── client.py      <- Query the API created by deployment
    │   │
    │   └── webservice  <- Scripts to manage Deployment of the model
    │       └── entry_script.py <- Used for deployment
    │
    ├── tests 
    │   ├── test_data.py       
    │   ├── test_datafetcher.py        
    │   ├── test_model.py     
    │   └── test_training.py 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    ├── requirements_tests.txt   <- The requirements file for reproducing the test environment
    │             
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
