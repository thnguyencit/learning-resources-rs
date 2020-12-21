# Code is to reproduce the experimental results
This code aims to reproduce the results in the paper **Deep Matrix Factorization for Learning Resources Recommendation**
# Prerequisites
Please install if you do not have: python3.6. Some packages you may need to install more.
```bash
pip install numpy
pip install matplotlib
pip install ConfigParser
pip install pandas
pip install sklearn
pip install tensorflow==1.14
pip install keras
pip install keras_sequential_ascii
```
## Usage

Please run this script to reproduce an experiment with the training set of **reviews_train** and validation set of **reviews_test**. These file are located in **at_least_5ratings** folder. This experiment run to 15 epochs.
```bash
python learning-resources-RS.py -t reviews_train -v reviews_test -p at_least_5ratings -e 15
```
Some parameters to run with command line:
- -p specify the path to file input
- -t : specify training set
- -v : specify validation set
- -e : specify the number of epochs
- -b : specific the number of batch size
- -s : if >0 show results by charts
