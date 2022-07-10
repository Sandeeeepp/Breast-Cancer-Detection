from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd

def DetectCancerSVM():
    cancer = datasets.load_breast_cancer()
    print("\n\nFeatures of cancer dataset : \n", cancer.feature_names)
    
    print("\n\nLabels of cancer dataset : \n", cancer.target_names)
    
    print("\n\nShape of dataset : \n", cancer.data.shape)
    
    print("---------First 5 records are : ----------")
    print(cancer.data[0:5])
    
    print("\nTarget of dataset : \n", cancer.target)

    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3)
    
    clf = svm.SVC(kernel = 'linear')
    
    clf.fit(X_train, Y_train)
    
    Y_pred = clf.predict(X_test)
    
    print("Accuracy of the model is : ", metrics.accuracy_score(Y_test, Y_pred) * 100)



def main():
    print("---------Breast Cancer Detection using SVM---------")
    DetectCancerSVM()

if __name__ == "__main__":
    main()