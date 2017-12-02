# Predicts duplicate questions in Quora using Deeplearning

* Use keras==1.2.2 and theano==0.8.2

You can simply run

```
KERAS_BACKEND=theano python deep_learning_duplicate_prediction.py
```

The script assumes that train.csv file (data given for training) is available in
the current directory. You can change this as well if you prefer..

## Known Issues

* If you see all the RAM (> 5-10 GB), most likely there is memory leakage..
  check your theano version..

* keras api related issues: The version 2.0 of keras has significant API changes
  from the version this code is expected.. Please install the relevant keras
  version...
