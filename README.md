# Epidemic Prediction 
Term project for ENSC413. An application of deep learning to help predict where an epidemic will occur next. 

## Installing the Software
If you haven't already, clone the repo as follows:
```
git clone git@github.com:brittanyhewitson/epidemic_prediction.git
cd epidemic_prediction
```

Next, create an environment. You can use conda as follows:
```
conda create -n epidemic_prediction python=3.5
conda activate epidemic_prediction
```

Alternatively, you can use venv:
```
python3 -m venv venv
source venv/bin/activate
```

Now install the requirements for the software:
```
pip install -r requirements.txt
```

Finally, run the setup file:
```
python3 setup.py develop
```

## Testing the Models with Various Parameters
To test the models with a variety of parameters, the `test_parameters.py` script can be used as follows:
```
python3 test_parameters.py --gpu --data_choice small_data
```
Where the `--gpu` flag indicates whether you wish to use a GPU and the `--data_choice` option specifies which data source to train on. The choices for data source are:
| Data Source | Description | Size |
| ----------- | ----------- | ---- |
| `big_data` | The complete raw input dataset. This file is large and imbalanced, therefore it will take several hours to train and will produce results of approximately 87% regardless of the parameters| 116109 rows |
| `small_data` | The small input dataset provided by the example article. This dataset is found in the `X.pkl` and `y.pkl` files and is relatively small compared to the large dataset. Therefore it will take approximately 5 minutes to run and should produce accuracies in the range of 80-95%| 1213 rows |
| `smote` | The complete input dataset upsampled using SMOTE. This dataset includes all rows from the input dataset, therefore holds data for all dates for all locations | 201948 rows |
| `by_date` | A number of CSV files, each representing the feature and output data for all locations on a single date. Only to be used to find the accuracy as a function of the date | 26 CSV files |
| `single_date` | Uses only a single CSV file from the list of CSV files in the `by_date` category | Size depends on which date is selected, varies from approximately 1000-2000 rows per file |

The default value for the data choice is a single date, with the date beign 2016-06-25. 

Running `test_parameters.py` will generate and save a series of plots relating to the accuracies produced by varying parameters for each model. 

## Testing the Models with Optimal Parameters
To test the models with optimal parameters, simply run the following:
```
python3 run.py
```
This will train and test the perceptron, multi-layer perceptron, SVM, k-NN, and random forest according to the parameters we have found that produce the best results. The accuracy for each classifier will be printed to the console, and a bar plot comparing the accuracies will be displayed and saved in a new directory called `output_images`. 
