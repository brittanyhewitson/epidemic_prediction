# Epidemic Prediction 
Term project for ENSC413. An application of deep learning to help predict where an epidemic will occur next. 

## Installing the Software
If you haven't already, clone the repo as follows:
```
git clone git@github.com:brittanyhewitson/epidemic_prediction.git
```

Next, create an environment. You can use conda as follows:
```
conda create -n epidemic_prediction python=3.5
conda activate epidemic_prediction
```

Alternatively, you can use virtualenv:
```
pip install virtualenv
virtualenv epidemic_prediction
source epidemic_prediction/bin/activate
```

Now install the requirements for the software:
```
cd epidemic_prediction
pip install -r requirements.txt
```

Finally, run the setup file:
python3 setup.py develop