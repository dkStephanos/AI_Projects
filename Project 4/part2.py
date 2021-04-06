import pandas as pd
from SVM import SVM

white_df = pd.read_csv('winequality-white.csv')

svm = SVM(C=.75, kernel='poly')
svm.set_data(white_df, 'quality')
X_train, X_test, y_train, y_test = svm.split_test_data(.3, True)
    
svm.fit_and_predict(X_train, X_test, y_train)
print(svm.get_classification_report(y_test))
svm.get_learning_curve()

red_df = pd.read_csv('winequality-red.csv')

svm = SVM(C=.75, kernel='poly')
svm.set_data(red_df, 'quality')
X_train, X_test, y_train, y_test = svm.split_test_data(.3, True)
    
svm.fit_and_predict(X_train, X_test, y_train)
print(svm.get_classification_report(y_test))
svm.get_learning_curve()
