from asyncio.windows_events import NULL
from distutils.log import error
from flask import Flask, flash, render_template, request
import pickle
import numpy as np
from pyparsing import Or
import sklearn.neighbors._base
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

prediction= -1


class ml:
    def svm(self,data):
      filename = 'svm.pkl'
      model = pickle.load(open(filename, 'rb'))
      prediction = model.predict(data)
      return prediction

    def nb(self,data):
         filename = 'nb.pkl'
         model = pickle.load(open(filename, 'rb'))
         prediction = model.predict(data)
         return prediction


    def logisticReg(self,data):
         filename = 'logisticRegression.pkl'
         model = pickle.load(open(filename, 'rb'))
         prediction = model.predict(data)
         return prediction   

    def randomF(self,data):
         filename = 'randomForest.pkl'
         model = pickle.load(open(filename, 'rb'))
         prediction = model.predict(data)
         return prediction


app=Flask(__name__)

app.config.update(
    TESTING=True,
    SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET','POST'])
def predict():
    prediction= -1
    if  request.method == "POST":

        age=int(request.form['Age'])
        gender=int(request.form['Gender'])
        cp=request.form.get('cp')
        trestbps=int(request.form['trestbps'])
        chol= int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach=int(request.form['thalach'])
        exang= request.form.get('exang')
        oldpeak=float(request.form['oldpeak'])
        slope=request.form.get('slope')
        ca=int(request.form['ca'])
        thal=request.form.get('thal')
        MLoption = request.form.get('MLmodel')
        data = np.array([[age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        models= ml()
        if(MLoption == 'svm'):
            prediction=models.svm(data)
            m="SVM model with accuracy 85%"
        elif MLoption== 'nb':
            prediction=models.nb(data)
            m="Naive Bayies model with accuracy 85%"
        elif MLoption =='rf':
            prediction=models.randomF(data)
            m="random Forest model with accuracy 85%"
        elif MLoption == 'lr':
            prediction=models.logisticReg(data)
            m="logistic regression model with accuracy 85%"


        return render_template('index.html',prediction=prediction,m=m)

if "__main__"== __name__:
    app.run(debug=False)  