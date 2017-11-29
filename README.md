This is an introductory guide on text generation using RNNs and Keras. Please follow the steps mentioned below.
**This tutorial will require Python3.x**

### Creating your environment
You can create your virtual environment using either Anaconda or Virtualenv. Follow the links for instructions on how to set up your environment
* [Virtualenv](https://virtualenv.pypa.io/en/stable/installation/)
* [Anaconda](https://conda.io/docs/user-guide/install/index.html)

**Make sure to create a python3.x environment** <br>
You can check to see that your environment is created with the correct python version by activating your environment and running the command `python --version`. You should see a version greater than 3.
### Steps to follow

* Clone the project into a local directory <br> `git clone https://github.com/kirit93/Personal.git`
* `cd text_generation_keras`
* Install the dependencies by running <br> `pip install -r requirements.txt`
* Run the notebook by running <br> `jupyter notebook`
* To run the code <br> For training - `python text_gen.py --mode train` <br> For testing - `python text_gen.py --mode test`. <br> While testing make sure to include the path to the weight file in the code.

