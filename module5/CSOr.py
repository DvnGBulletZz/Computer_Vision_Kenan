import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)

# datalength = len(digits)
samples = len(digits.images)
data =  digits.images.reshape((samples, -1))




# print(data)


clf = svm.SVC(gamma=0.001, C=100)
# X,y = digits.data[:-10], digits.target[:-10]


X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.33)    #verdeel data over 2 sets randomly
clf.fit(X_train,y_train)
# print(y_test)

# print(clf.predict(digits.data[-4:-3])) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit

prediction = clf.predict(X_test)
# precentage = 100/len(prediction)
# print(precentage)
count = 0
count_wrong = 0

for row in range(len(y_test)):

    if y_test[row] == prediction[row]:
        count += 1
    else:
        count_wrong +=1
     

        
# result = precentage * count



print ("ACCURACY:", accuracy_score(y_test, prediction)*100)
print("amount right",count ,"amount wrong", count_wrong )

print("total amount of data", len(X_test) + len(X_train))
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()