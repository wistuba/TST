# Two-Stage Transfer Surrogate Model
We provide here the source code for our paper ["Two-Stage Transfer Surrogate Model for Automatic Hyperparameter Optimization"](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf).

## Usage
The class SMBOMain.java contains the main function.
Executing the program without parameters will print a help describing all parameters.
Following line executes SMBO for 10 trials using TST-R on the data set A9A using the meta-data set from all remaining 49 other data sets.
```
java SMBOMain -f data/svm/ -dataset A9A -tries 10 -s tst-r -bandwidth 0.1 -hpRange 6 -hpIndicatorRange 3
```

## Meta-Data
The two meta-data sets used in our experiments are available in the folder "data". More information is available on our project website. Visualizations of the SVM meta-data set can be found [here](http://www.hylap.org/meta_data/svm/) and the creation of the Weka meta-data set can be found [here](http://www.hylap.org/meta_data/weka/).

## Dependencies
Our code makes use of [Apache Commons Math](https://commons.apache.org/proper/commons-math/). The library is provided in the folder "lib".

## Cite
```
@INPROCEEDINGS{WistubaECML2016,
  author    = {Martin Wistuba and
               Nicolas Schilling and
               Lars Schmidt{-}Thieme},
  title     = {Two-Stage Transfer Surrogate Model for Automatic Hyperparameter Optimization},
  booktitle = {Machine Learning and Knowledge Discovery in Databases - European Conference,
               {ECML} {PKDD} 2016, Riva del Garda, Italy, September 19-23, 2016,
               Proceedings, Part {I}},
  pages     = {199--214},
  year      = {2016}
}
```
