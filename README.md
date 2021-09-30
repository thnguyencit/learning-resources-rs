# Code is to reproduce the experimental results
This code aims to reproduce the results in the paper **Deep Matrix Factorization for Learning Resources Recommendation**

Please cite this work, if this can help you:

Dien T.T., Thanh-Hai N., Nghe N.T. (2021) Deep Matrix Factorization for Learning Resources Recommendation. In: Nguyen N.T., Iliadis L., Maglogiannis I., Trawiński B. (eds) Computational Collective Intelligence. ICCCI 2021. Lecture Notes in Computer Science, vol 12876. Springer, Cham. https://doi.org/10.1007/978-3-030-88081-1_13

```
Dien T.T., Thanh-Hai N., Nghe N.T. (2021) Deep Matrix Factorization for Learning Resources Recommendation. In: Nguyen N.T., Iliadis L., Maglogiannis I., Trawiński B. (eds) Computational Collective Intelligence. ICCCI 2021. Lecture Notes in Computer Science, vol 12876. Springer, Cham. https://doi.org/10.1007/978-3-030-88081-1_13
```


```
@incollection{Dien2021,
  doi = {10.1007/978-3-030-88081-1_13},
  url = {https://doi.org/10.1007/978-3-030-88081-1_13},
  year = {2021},
  publisher = {Springer International Publishing},
  pages = {167--179},
  author = {Tran Thanh Dien and Nguyen Thanh-Hai and Nguyen Thai Nghe},
  title = {Deep Matrix Factorization for Learning Resources Recommendation},
  booktitle = {Computational Collective Intelligence}
}
```

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

Please run this script to reproduce an experiment with the training set of **reviews_train** and validation set of **reviews_test**. These file are located in **at_least_5ratings** folder. This experiment can run to 15 epochs.
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
Personal site: https://sites.google.com/site/ntnghe

Email: ntnghe@cit.ctu.edu.vn

Domain of Research: Intelligent Systems,Artificial Intelligence,e-Business,e-Learning,Artificial Intelligence,Big Data,Business Intelligence,Computational Intelligence,Computer Vision,Data Mining,Data Retrieval and Data Mining,Data Science,Databases,Decision Support Systems,e-Business,Educational Systems Design,e-Learning,Expert Systems,Information Retrieval,Intelligent Systems,Knowledge Engineering and Mining,Knowledge Management,Machine Learning,Mobile Applications,Soft Computing

Biography: Dr. NGUYEN THAI-NGHE is an Associate Professor of the department of Information Systems at Can Tho University. He received the B.S. degree in Informatics from Can Tho University (CTU) and the M.S. degree in Information Management from Asian Institute of Technology, Thailand. In 2009-2012, he got the scholarship of WorldBank-CTU to do Ph.D. degree in Computer Science at University of Hildesheim, Germany. He is the PC member/reviewer of several international conference and journal such as Springer FDSE, IEEE ACOMP, IEEE KSE, Springer ACIIDS, PLOS ONE, ASTESJ, etc.
### Nguyen Thanh-Hai
Email: nthai@cit.ctu.edu.vn

Domain of Research: Artificial Intelligence, Bioinformatics, Computer Vision,Databases, Health care Systems, Machine Learning, Recommendation Systems, Simulation Systems

Biography: He is a lecturer of College of Information and Communication Technology, Can Tho University, Vietnam. He received his Engineering degree in Informatics from Can Tho University, the master degree in Computer Science and Engineering from National Chiao Tung University, Taiwan, and obtained the PhD degree in Computer Science from Sorbonne University, France. His PhD thesis studied approaches for disease prediction using metagenomic data. His current research includes Bioinformatics, Health care system, Computer Vision, modeling of decisions, and recommender system.
### Tran Thanh Dien
Personal site: https://sites.google.com/a/ctu.edu.vn/thanhdien/

Email: thanhdien@ctu.edu.vn

Biography: He is a lecturer of College of Information and Communication Technology, Can Tho University (CTU), Vietnam. He received the B.S. degree in Informatics from Can Tho University in 1998, the MEng. degree in Information Management from Asian Institute of Technology (AIT), Thailand in 2006. From August 2017 up to now, he is PhD candidate in Information Systems of Can Tho University, Vietnam.

Research topics: Recommender Systems, Machine Learning, Ontologies in education


