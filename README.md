# Two-Stage Transfer Surrogate Model
We provide here the source code for our paper "Two-Stage Transfer Surrogate Model for Automatic Hyperparameter Optimization".

##Usage
The class SMBOMain.java contains the main function.
Executing the program without parameters will print a help describing all parameters.
Following line is executes SMBO for 10 trials using TST-R on the data set A9A using the meta-data set from all remaining 49 other data sets.
```
java SMBOMain -f data/svm/ -dataset A9A -tries 10 -s tst-r -bandwidth 0.1 -hpRange 6 -hpIndicatorRange 3
```

##Meta-Data
The two meta-data sets used in our experiments are available in the folder "data". More information is available on our project website. Visualizations of the SVM meta-data set can be found [here](http://www.hylap.org/meta_data/svm/) and the creation of the Weka meta-data set can be found [here](http://www.hylap.org/meta_data/weka/).
