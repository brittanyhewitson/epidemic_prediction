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

Alternatively, you can use virtualenv:
```
python3 -m venv venv
source venv/bin/activate
```

Now install the requirements for the software:
```
pip install -r requirements.txt
```

Finally, run the setup file:
python3 setup.py develop