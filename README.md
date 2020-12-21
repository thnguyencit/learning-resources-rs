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

## Authors

### Nguyen Thai-Nghe
Domain of Research: Intelligent Systems,Artificial Intelligence,e-Business,e-Learning,Artificial Intelligence,Big Data,Business Intelligence,Computational Intelligence,Computer Vision,Data Mining,Data Retrieval and Data Mining,Data Science,Databases,Decision Support Systems,e-Business,Educational Systems Design,e-Learning,Expert Systems,Information Retrieval,Intelligent Systems,Knowledge Engineering and Mining,Knowledge Management,Machine Learning,Mobile Applications,Soft Computing

Biography: Dr. NGUYEN THAI-NGHE is an Associate Professor of the department of Information Systems at Cantho University. He received the B.S. degree in Informatics from Cantho University (CTU) and the M.S. degree in Information Management from Asian Institute of Technology, Thailand. In 2009-2012, he got the scholarship of WorldBank-CTU to do Ph.D. degree in Computer Science at University of Hildesheim, Germany. He is the PC member/reviewer of several international conference and journal such as Springer FDSE, IEEE ACOMP, IEEE KSE, Springer ACIIDS, PLOS ONE, ASTESJ, etc.
