# Create your views here.
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse,HttpResponseRedirect
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from mlxtend.plotting import category_scatter
from sklearn.metrics import confusion_matrix 
from mlxtend.plotting import plot_confusion_matrix
from pylab import *
import base64,urllib,io
from io import StringIO, BytesIO
import importlib


#matplotlib = importlib.reload(matplotlib)
#matplotlib.use('cairo')
import matplotlib.pyplot as plt,mpld3



# Create your views here.

from .forms import InputForm
#dataframe = pandas.read_csv('static/pima_indians_diabetes.csv')

def get_text(request):
#    dataframe = pandas.read_csv('static/pima_indians_diabetes.csv')
    #print(dataframe.head())
    if request.method == 'POST':
        form = InputForm(request.POST)
        request.session['arr']=[]
        
        if form.is_valid():
           # a1=request.session['Text'] = form.cleaned_data['Text']
            #arr.append(int(a1))
            request.session['Pregnancies'] = int(form.cleaned_data['Pregnancies'])
        #    arr.append(int(a1))
            request.session['Glucose'] = int(form.cleaned_data['Glucose'])
         #   arr.append(int(a2))
            request.session['BloodPressure'] = int(form.cleaned_data['BloodPressure'])
          #  arr.append(int(a3))
            request.session['SkinThickness'] = int(form.cleaned_data['SkinThickness'])
           # arr.append(int(a4))
            #print(arr)            
            request.session['Insulin'] = int(form.cleaned_data['Insulin'])
            #arr.append(int(a5))
            request.session['BMI'] = float(form.cleaned_data['BMI'])
            #a6=float(a6)
            request.session['DiabetesPedigreeFunction'] = float(form.cleaned_data['DiabetesPedigreeFunction'])
            #a7=float(a7)
            request.session['Age'] = int(form.cleaned_data['Age'])
            #arr.append(int(a8))
            #print(arr)
            request.session['arr'].append([request.session['Pregnancies'],request.session['Glucose'],
                request.session['BloodPressure'],request.session['SkinThickness'],request.session['Insulin'],
                request.session['BMI'],request.session['DiabetesPedigreeFunction'],request.session['Age']])
            print(request.session['arr'])

            return HttpResponseRedirect('/result/')
    else:
        form = InputForm()

    return render(request,'input.html', {'form': form})
'''

======
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
=====
'''


#===================================================
def result(request):
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv('static/pima_indians_diabetes.csv', names=names)

 #   print(dataframe.head()) 
#    print(dataframe.isnull().sum())

    array = dataframe.values
    X = array[:,0:8]
    Y = array[:,8]

    Xc=dataframe[['plas','age','class']]
    Xc=Xc.astype(int)

    Xc=np.array(Xc)
    fix = category_scatter(x=0, y=1, label_col=2,data=Xc, legend_loc='upper left')
    #plt.figure(figsize=(20,10))
    plt.show()
#    grid(True)
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img1 = urllib.parse.quote(string)
    #print(uri)
    

# Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 2)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state =72)
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)
    K_value = 15
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='kd_tree')#15
    neigh.fit(X_train, y_train) 
    y_pred = neigh.predict(X_test)

    y_pred2= neigh.predict(request.session['arr'])
    y_pred1 = neigh.predict([[1,85,66,29,0,26.6,0.351,31],[6,148,72,35,0,33.6,0.627,50],[1,85,66,29,0,26.6,0.351,31],
            [0,137,40,35,168,43.1,2.288,33]])
    print('its', y_pred2)
    value=(y_pred2[0])

    if value==0 :
        test_result='Negative'
    else:
        test_result='Positive'

    print(test_result)

    print(value)
    print(y_pred1)
    res=accuracy_score(y_test,y_pred)*100
    print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
    cm = confusion_matrix(y_test, y_pred)

     
    fig, ax = plot_confusion_matrix(conf_mat=cm,figsize=(3,3.5))

    #plt.set_size_inches(8.5, 4.5)
  #  grid(True)
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img2 = urllib.parse.quote(string)


    X = np.array(dataframe[['plas','age']])
    Y1=Y.astype(int)
    y = np.array(Y1)
    neigh.fit(X, y)


# Plotting decision regions
    plt.figure(figsize=(13,7))
    
    plot_decision_regions(X, y, clf=neigh, legend=2)

# Adding axes annotations
    plt.xlabel('Plasma glucose')
    plt.ylabel('Age')
    plt.title('knn classification')
    #grid(True)
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    img3 = urllib.parse.quote(string)




#    return HttpResponse("Thanks")    
    return render(request,'result.html', {'res':res,'test_result':test_result,
        "img1":img1,"img2":img2,"img3":img3})


def dataset(request):
    return render(request,'dataset.html',{})
def algorithm(request):
    return render(request,'algo.html',{})


def getimage(request):
    # Construct the graph
    x = arange(0, 2*pi, 0.01)
    s = cos(x)**2
    plot(x, s)

    xlabel('xlabel(X)')
    ylabel('ylabel(Y)')
    title('Simple Graph!')
    grid(True)
    '''
    fig = plt.figure()
    plt.plot([3,1,4,1,5])
    figHtml = mpld3.fig_to_html(fig)
    result = {'fileData': myData, 'figure': figHtml}
    #return render_to_response("index.html",result)
    
    # Store image in a string buffer
    out = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(out, "PNG")
    pylab.close()
    '''


    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = figfile.read()  # extract string
    import base64,urllib,io
    figdata_png = base64.b64encode(figdata_png)
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
#    print(uri)
    
    #uri = 'data:image/png;base64,' + urllib.parse.quote(figdata_png)
    #html = '<img src = "%s"/>' % uri

    return render(request,'img.html',{'uri':uri,'figdata_png':figdata_png},result)
    # Send buffer in a http response the the browser with the mime type image/png set
    #return HttpResponse(out.getvalue(), content_type="image/png")


