# Advanced Datamining
**Bio-Informatics Year 3, Period 11 (2022-2023)**

Learning about and understanding the basic concepts of neural networks by creating one yourself from scratch.


## About the course assignments
The course is taught in 3 parts, each with two chapters, from the basics gradually towards creating a complete basal neural network. 
For every chapter a syllabus with theory is supplied and a [Jupyter Notebook](https://jupyter.org/) teaching how to apply the learned theory by coding it yourself.

At the end of the course understanding about neural networks for classification and regression should be acquired.
Knowledge about the workings of neural networks should be possessed, terms like 
* (multi-layer) perceptron, 
* cost-function and cross-entropy, 
* (stochastic) gradient descent, 
* back-propagation, 
* soft-max, 
* adaptive learning, 
* multiple forms of regularisation and data-augmentation,
* over- and underfitting.

The course is graded by handing in a final assignment and by doing an oral examination about neural network theory.
The final assignment consists out of two parts, the first part is creating your final neural network Python module 
and the second part is to use that module to create a model for a real dataset.

### My own spin on the course; vlearning package
The way every chapter had the same `data.py` module and a different version of the `model.py` module,
was not to my liking, because it was not very organised with a LOT of duplicate code.
This is why I decided to create a Python package instead, that will contain all the code for the models nicely separated in different modules.
I decided to name this package `vlearning`, which I found appropriate and funny to sort of include my name in it, since it stands for "Vincent's Learning".


## Repository file structure
```
Advanced-Datamining
├── .gitignore
├── environment.yml
├── LICENSE
├── README.md
├── Chapter01
│   └── *
├── Chapter02
│   └── *
├── Chapter03
│   └── *
├── Chapter04
│   └── *
├── Chapter05
│   └── *
├── Final-Assignment
│   └── *
└── vlearning
    ├── pyproject.toml
    ├── README.md
    └── vlearning
        └── *
```


## Installation
This project is made on a Unix-like system (Linux/macOS) but through the use of conda environments it should also work fine on a Windows system.

### Conda/Mambaforge
To manage the virtual environment and dependencies the [Conda](https://conda.io/)-based Python3 distribution [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) is used.
Mambaforge provides the required Python and Conda commands, but also includes Mamba, an extremely fast and robust replacement for the Conda package manager. 
Since the default conda solver is large, slow and sometimes has issues with selecting the latest package releases.

Download the latest installer script of Mambaforge for your OS from https://github.com/conda-forge/miniforge#mambaforge and follow the instructions listed there to install it.

### Create environment
The next thing needing to be done before being able to use the models is to create the environment with all the required dependencies. 
This is easily done using Mamba and the `environment.yml` file included in this repository.

Open a terminal and ensure your terminal has the base mamba environment activated with
```bash
mamba activate base
```
Make sure the working directory is the root of this repository and then simply use the following command to create the environment:
```bash
mamba env create
```
This will name the environment `advanced-datamining`, if desired it can be given another name by adding `--name your-desired-name`.

### Install `vlearning` python package
All the Python code written for this course is located in the `vlearning` package.
It is thus required to install this package in the environment to be able to run the Jupyter Notebooks for the chapters or the final assignment.
First we'll activate the just-created environment and after that we'll install the package using [Poetry](https://python-poetry.org/), which was specified in the environment.yml so it should already be installed.
Use the following commands to activate the environment and install the package:
```bash
mamba activate advanced-datamining
cd vlearning
poetry install
```


## Usage
To run a model open the terminal with the working directory of the pipeline you want to run.  
Activate the environment by running the following command (if your environment is named differently replace `advanced-datamining` with that name):
```bash
mamba activate advanced-datamining
```
After the environment is activated a [jupyter notebook](https://jupyter.org/) should be launched, after running the command it should automatically open your webbrowser. 
In the webbrowser is a file browser where you can select one of the chapter directories and open the notebook of that selected chapter.
```bash
jupyter notebook
```
In the notebook you can run all the code chunks, or just a selection until a certain point. 
But make sure that if for example you want to run the third chunk you also run the two chunks before that! 
Since it might use object create before the actual chunk you want to run.

### Converting notebook to PDF
To convert the Jupyter Notebook to pdf simply use the following command:
```bash
jupyter nbconvert --to webpdf --allow-chromium-download path/to/notebook.ipynb 
```


## Useful links
* [Conda documentation](https://docs.conda.io/projects/conda/en/latest)
* [Jupyter documentation](https://docs.jupyter.org/en/latest/)


## Contact
For support or any other questions, do not hesitate to contact me at v.k.talen@st.hanze.nl
