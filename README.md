# Code is reproduce the experimental results in the paper "Deep Matrix Factorization for Learning Resources Recommendation"
# Prerequisites
Please install if you do not have: python3.6
```bash
pip install numpy
pip install matplotlib
pip install ConfigParser
pip install pandas
pip install sklearn
pip install tensorflow==1.14
pip install keras
pip install keras_sequential_ascii
pip install minisom
pip install pillow
```
## Usage

Please run this code to reproduce the experiment
```bash
python learning-resources-RS.py -t reviews_train -v reviews_test -p at_least_5ratings -e 15
```
With:
- -p specify the path to file input
- -t : specify training set
- -v : specify validation set
- -v : specify the number of epochs
