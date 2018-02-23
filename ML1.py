# Check the versions of libraries

# Python version
import sys
#print('Python: {}'.format(sys.version))
# scipy
import scipy
#print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
#print('numpy: {}'.format(numpy.__version__))
# mexitatplotlib
import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
#print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))
#import pandas
import pandas
print("Importing all Packages done!")
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
print('Importing all classifiers and plots done!')

#Load Dataset
print("File Opening status: ",end="")

# irisDataset = open('C:\Program Files\Weka-3-8\data\iris.arff','r')
# C:\\MyData\\Coding\\WEKA\\iris.txt
# C:\\MyData\\Coding\\WEKA\\iris.arff
# C:\\Coding\\pandas\\pandas\\tests\\data\\iris.csv
url1 = 'C:\\MyData\\Coding\\WEKA\\iris.csv'
print("Done!")

#Attributes of dataset
names = ['sepal-length','sepal-width','petal-length','petal-width','class']

#dataset stored
dataset = pandas.read_csv(url1, names=names)
print('Dataset is initialised, Done!')

# shape
def shaper():
    print(dataset.shape)

# head
def header():
    print(dataset.head(20))

# Description of out datasets group by their attributes
# This includes the count, mean, the min and max values as well as some percentiles.
def describer():
    print(dataset.describe())

# Class Description
def descriptioner():
    print(dataset.groupby('class').size())

# Unvariat Plot
# box and whisker plots
def boxploter():
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

# Creating Histogram
def histogramer():
    dataset.hist()
    plt.show()

# Creating Multivariate plot
def multivariater():
    scatter_matrix(dataset)
    plt.show()

#Selection will be go through this code
print("Enter your choise : ",end="")
val = input()
if val == "1" :
    shaper();header(); describer();  descriptioner() 
elif val == "2":
    boxploter()
elif val == "3":
    histogramer()
elif val == "4":
    multivariater()
else :
    print("enter daf")
