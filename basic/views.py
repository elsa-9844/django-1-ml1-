from django.shortcuts import render

# Create your views here.
def knn(request):
    if(request.method=="POST"):
            dat=request.POST
            a=float(dat.get('aa'))
            e=float(dat.get('bb'))
            sa=dat.get('male')
            sb=dat.get('female')
            bpa=dat.get('high')
            bpb=dat.get('low')
            bpc=dat.get('nom')
            ca=dat.get('hi')
            cb=dat.get('n1')
            if(sa):
                 b=1
            elif(sb):
                 b=0
            if(bpa):
                 c=0
            elif(bpb):
                 c=1
            elif(bpc):
                 c=2
            if(ca):
                 d=0
            elif(cb):
                 d=1
            if('submit' in request.POST):
                 import pandas as pd
                 import sklearn
                 from sklearn.model_selection import train_test_split
                 from sklearn.neighbors import KNeighborsClassifier
                 from sklearn.metrics import confusion_matrix
                 from sklearn.metrics import accuracy_score
                 from sklearn.preprocessing import LabelEncoder
                 le=LabelEncoder()
                 path="C:\\Users\\dell8\\OneDrive\\Documents\\onedrive\\Desktop\\folder\\project\\project1\\drug200.csv"
                 data = pd.read_csv(path)
                 data['Sex'] = le.fit_transform(data['Sex'])
                 data['BP'] = le.fit_transform(data['BP'])
                 data['Cholesterol'] = le.fit_transform(data['Cholesterol'])
                 inputs = data.drop(['Drug'],axis=1)
                 outputs = data.drop(['Age','Sex','BP','Cholesterol','Na_to_K'],axis=1)
                 x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
                 model=KNeighborsClassifier(n_neighbors=13)
                 model.fit(x_train,y_train)
                 y_pred=model.predict(x_test)
                 result = model.predict([[a,b,c,d,e]])
                 accuracy = accuracy_score(y_test, y_pred)
                 acc=accuracy*100
                 return render(request, 'knn.html',context={'result':result,'acc':acc})
    return render(request,"knn.html")

def rforest(request):
    if(request.method=="POST"):
            dat=request.POST
            a=float(dat.get('aa'))
            e=float(dat.get('bb'))
            sa=dat.get('male')
            sb=dat.get('female')
            bpa=dat.get('high')
            bpb=dat.get('low')
            bpc=dat.get('nom')
            ca=dat.get('hi')
            cb=dat.get('n1')
            if(sa):
                 b=1
            elif(sb):
                 b=0
            if(bpa):
                 c=0
            elif(bpb):
                 c=1
            elif(bpc):
                 c=2
            if(ca):
                 d=0
            elif(cb):
                 d=1
            if('submit' in request.POST):
                 import pandas as pd
                 import sklearn
                 from sklearn.model_selection import train_test_split
                 from sklearn.preprocessing import LabelEncoder
                 from sklearn.ensemble import RandomForestClassifier
                 from sklearn.metrics import accuracy_score
                 le=LabelEncoder()
                 path="C:\\Users\\dell8\\OneDrive\\Documents\\onedrive\\Desktop\\folder\\project\\project1\\drug200.csv"
                 data = pd.read_csv(path)
                 data['Sex'] = le.fit_transform(data['Sex'])
                 data['BP'] = le.fit_transform(data['BP'])
                 data['Cholesterol'] = le.fit_transform(data['Cholesterol'])
                 inputs = data.drop(['Drug'],axis=1)
                 outputs = data.drop(['Age','Sex','BP','Cholesterol','Na_to_K'],axis=1)
                 x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
                 model = RandomForestClassifier(n_estimators=200)
                 model.fit(x_train,y_train)
                 y_pred=model.predict(x_test)
                 result = model.predict([[a,b,c,d,e]])
                 accuracy = accuracy_score(y_test, y_pred)
                 acc=accuracy*100
                 return render(request, 'rforest.html',context={'result':result,'acc':acc})
    return render(request,"rforest.html")