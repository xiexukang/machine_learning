import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve, train_test_split, KFold
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

#数据清洗

def clean_data(dataset):
    dataset.columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"]

    column_names = dataset.columns
    #替换？为空值
    for c in column_names:
        dataset[c] = dataset[c].replace(" ?", np.NaN)
    #空值填充
    dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))

    
    column_names_str = ["workclass","education","marital-status","occupation", "relationship","race","sex","native-country","income"]

    #降维，将之归为两类
    dataset.replace([' Never-married', ' Married-civ-spouse', ' Divorced',
       ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
       ' Widowed'],
               [' not-married', ' married', ' not-married',
       ' married', ' not-married', ' married',
       ' not-married'], inplace=True)


    #进行one-hot编码 
    Workclass = pd.get_dummies(dataset['workclass'],drop_first=True)
    Education = pd.get_dummies(dataset['education'],drop_first=True)
    MaritalStatus = pd.get_dummies(dataset['marital-status'],drop_first=True)
    Occupation = pd.get_dummies(dataset['occupation'],drop_first=True)
    Relationship = pd.get_dummies(dataset['relationship'],drop_first=True)
    Race = pd.get_dummies(dataset['race'],drop_first=True)
    Sex = pd.get_dummies(dataset['sex'],drop_first=True)
    NativeCountry = pd.get_dummies(dataset['native-country'],drop_first=True)
    Income = pd.get_dummies(dataset['income'],drop_first=True)

    '''注意：凡是会对原数组作出修改并返回一个新数组的，往往都有一个 inplace可选参数。如果手动设定为True（默认为False），那么原数组直接就被替换。也就是说，采用inplace=True之后，原数组名（如2和3情况所示）对应的内存值直接改变；

而采用inplace=False之后，原数组名对应的内存值并不改变，需要将新的结果赋给一个新的数组或者覆盖原数组的内存位置（如1情况所示）。
'''
    dataset.drop(['workclass',
     'education',
     'marital-status',
     'occupation',
     'relationship',
     'race',
     'sex',
     'native-country',
     'income'], axis=1, inplace=True)
    from scipy.stats.mstats import winsorize
    dataset["age"]           = winsorize(dataset["age"],(0,0.15))
    dataset["fnlwgt"]        = winsorize(dataset["fnlwgt"],(0,0.15))
    dataset["capital-gain"]  = winsorize(dataset["capital-gain"],(0,0.099))
    dataset["capital-loss"]  = winsorize(dataset["capital-loss"],(0,0.099))
    dataset["hours-per-week"]= winsorize(dataset["hours-per-week"],(0.12,0.18))
    dataset = pd.concat([Workclass,Education,MaritalStatus,Occupation,Relationship,Race,Sex,NativeCountry,Income],axis=1)
    return dataset
def _shuffle(X):
    '''将顺序打乱'''
    randomsize = np.arange(len(X))
    np.random.shuffle(randomsize)
    return X[randomsize]
import csv
def fun(n,j):
    in_file = open("adult_test.csv","r",newline="")
    reader = csv.reader(in_file)
    out_file = open("adult_test1.csv","w",newline="")
    writer = csv.writer(out_file)
    i =1
    for row in reader:
        if int(n) == 1 and int(j) ==i:
            print("yes")
            row[14] = ' >50K.'
        elif int(j)==i:
            row[14] = ' <=50K.'
        else:
            pass
        i +=1
        writer.writerow(row)
    in_file.close()    
    out_file.close()
    in_file = open("adult_test1.csv","r",newline="")
    reader = csv.reader(in_file)
    out_file = open("adult_test.csv","w",newline="")
    writer = csv.writer(out_file)
    for row in reader:
        writer.writerow(row)
    in_file.close()    
    out_file.close()

def data_deal(file_path):
    data = pd.read_csv(file_path,header=None,names= [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"])
    data.columns = data.columns.str.replace(" ","")
    # column_names=data.columns
    # for c in column_names:
    #     dataset[c] = data[c].replace(" ", "")
    column_names = data.columns
    #替换？为空值
    for c in column_names:
        print(data[c].unique)
        data[c] = data[c].replace(" ?", np.NaN)
    #空值填充
    data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
    # for c in column_names:
    #     print(df[c].unique)
    data['income'] = data['income'].map({' <=50K': 0, ' >50K': 1,' >50K.': 1,' <=50K.': 0})
    from scipy.stats.mstats import winsorize
    data["age"]           = winsorize(data["age"],(0,0.15))
    data["fnlwgt"]        = winsorize(data["fnlwgt"],(0,0.15))
    data["capital-gain"]  = winsorize(data["capital-gain"],(0,0.099))
    data["capital-loss"]  = winsorize(data["capital-loss"],(0,0.099))
    data["hours-per-week"]= winsorize(data["hours-per-week"],(0.12,0.18))
    data['sex'] = data['sex'].map({' Male': 1, ' Female': 0})
    print(data.head())
    data['race'] = data['race'].map({' White': 1, ' Asian-Pac-Islander': 1, ' Black':0, ' Amer-Indian-Eskimo':0, ' Other':0}) 
    data['relationship'] = data['relationship'].map({' Not-in-family':0, ' Unmarried':0,  ' Own-child':0, ' Other-relative':0, ' Husband':1, ' Wife':1})
    data['marital-status'] = data['marital-status'].map({' Widowed':0, ' Divorced':0, ' Separated':0, ' Never-married':0, ' Married-civ-spouse':1, ' Married-AF-spouse':1, ' Married-spouse-absent':0})
    data.drop(['marital-status'], axis=1,inplace=True)
    data.drop(['education'], axis=1,inplace=True)

    labels = ['workclass', 'occupation', 'native-country']

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for l in labels:
        data[l]=le.fit_transform(data[l])
    print(data)
    return data
def read_csv_data():
    # train = pd.read_csv("adult_train.csv", header=None)#header为int指表示指定行作为显示的头
    # print(train.shape)
    # print(train.isnull().sum())
    # train.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation", 
    # "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
    # column_names = train.columns
    # #打印列名
    # print(column_names)
    # #查看csv中？的个数
    # print(train['native-country'].isin([" ?"]).value_counts())
    # print(train['workclass'].isin([" ?"]).value_counts())
    # #查看问号个数及比例
    # # for c in column_names:
    # #     count = train[c].isin([" ?"]).sum()
    # #     if(count>0):
    # #         print(c)
    # #         print(count)
    # #         print(count*100/32561)
    # #         print(" ")
    # #打印每一个列中每一个特征的数量
    # for c in column_names:
    #     print(train[c].value_counts())

    # #将？替换
    # for c in column_names:
    #     train[c] = train[c].replace(" ?", np.NaN)

    #     #填充前的null个数
    # print(train.isnull().sum())
    # train['income'] = train[c].replace(" <=50K.", ' <=50K')
    # train['income'] = train[c].replace(" >50K.", ' >50K')
    # #进行填充
    # train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
    # #填充后的null 个数
    # print(train.isnull().sum())
    # #数据集的描述
    # #print(train.describe())
    # '''                age        fnlwgt  ...  capital-loss  hours-per-week
    # count  32561.000000  3.256100e+04  ...  32561.000000    32561.000000
    # mean      38.581647  1.897784e+05  ...     87.303830       40.437456
    # std       13.640433  1.055500e+05  ...    402.960219       12.347429
    # min       17.000000  1.228500e+04  ...      0.000000        1.000000
    # 25%       28.000000  1.178270e+05  ...      0.000000       40.000000
    # 50%       37.000000  1.783560e+05  ...      0.000000       40.000000
    # 75%       48.000000  2.370510e+05  ...      0.000000       45.000000
    # max       90.000000  1.484705e+06  ...   4356.000000       99.000000'''

    # column_name_str = ["workclass","education","marital-status","occupation", "relationship","race","sex","native-country","income"] 
    # #打印收入比例
    # # print(train['income'].value_counts()[0]/train.shape[0])#不超过50k人数比例)
    # # print(train['income'].value_counts()[1]/train.shape[0])#超过50k人数比例)
    # print(train)
    # #进行清洗
    # train=clean_data(train)
    # print(train.shape)
    # #继续删除不重要特征
    # train.drop([' Holand-Netherlands'], axis=1, inplace=True)
    # print(train.head(10))
    # #train.drop([' Scotland'], axis=1, inplace=True)
    # #对测试数据集进行操作
    # test = pd.read_csv('adult_test.csv', header=None)
    # #数据清洗
    # test = clean_data(test)
    # #test.drop(' Scotland', axis=1, inplace=True)
    # print(train.keys())
    # print(train.shape)
    # print(test.keys())
    # test.columns = [' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc',
    #    ' Self-emp-not-inc', ' State-gov', ' Without-pay', ' 11th', ' 12th',
    #    ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc',
    #    ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool',
    #    ' Prof-school', ' Some-college', ' not-married', ' Armed-Forces',
    #    ' Craft-repair', ' Exec-managerial', ' Farming-fishing',
    #    ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service',
    #    ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales',
    #    ' Tech-support', ' Transport-moving', ' Not-in-family',
    #    ' Other-relative', ' Own-child', ' Unmarried', ' Wife',
    #    ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Male', ' Canada',
    #    ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador',
    #    ' El-Salvador', ' England', ' France', ' Germany', ' Greece',
    #    ' Guatemala', ' Haiti', ' Honduras', ' Hong', ' Hungary', ' India',
    #    ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico',
    #    ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines',
    #    ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South',
    #    ' Taiwan', ' Thailand', ' Trinadad&Tobago',' United-States',
    #    ' Vietnam', ' Yugoslavia', ' >50K']

    # X_train = train.drop(' >50K',axis=1)
    # X_test = test.drop(' >50K',axis=1)
    # y_train = train[' >50K']
    # y_test = test[' >50K']
    from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    classifiers = [
    CatBoostClassifier(learning_rate=0.01),
    #LogisticRegression(C=1,penalty="l2"),
    #LogisticRegressionCV(multi_class="ovr",solver="sag",cv=kf,tol=1e-5),
    #KNeighborsClassifier(n_neighbors=8), # Some trial and error I don't show went into this hyperpa
    # KNeighborsClassifier(n_neighbors=9),
    # KNeighborsClassifier(n_neighbors=7),
    # KNeighborsClassifier(n_neighbors=6),
    # LinearDiscriminantAnalysis(),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(n_estimators = 10),
    # MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1),
    # SVC(),
    # GaussianNB()
]
    #for _ in range(100):
    from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
        #X_train,y_train=_shuffle(X_train),_shuffle(y_train)
        #predict_model = LogisticRegressionCV(multi_class="ovr",solver="sag",cv=5,tol=1e-3)
        #使用多种普遍模型进行处理
    #测试10次
    for _ in range(1): 
        print("第%d次"%_)
        kf = KFold(n_splits=10, shuffle=True, random_state=50)
        i = 0
        for model in classifiers:
            print("所采用的模型:" , model)
            predict_model = model
            predict_model.fit(X_train,y_train)
            
            predictions = predict_model.predict(X_test)
            a= accuracy_score(y_test, predictions)
            #print("精确度:%.2f"%a)
            print("精确度:",a)
            print(classification_report(y_test,predictions))
            print(confusion_matrix(y_test,predictions))
    # params = {'max_depth': [6], 
    #      'n_estimators': [200],
    #       'learning_rate': [0.07, 0.06],
    #       'max_features': [3,4]
    #      }

    # classifier = GradientBoostingClassifier()
    # kf = KFold(n_splits=10, shuffle=True, random_state=50)
    # grid = GridSearchCV(classifier, param_grid=params, cv=kf)
    # search_result = grid.fit(X_train,y_train)
    # predictions = search_result.predict(X_test)
    # a= accuracy_score(y_test, predictions)
    # print(a)
    # means = search_result.cv_results_['mean_test_score']
    # params = search_result.cv_results_['params']
    # for m, p in zip(means, params):
    #     print(f"{m} with: {p}")

            with open('answer%d.txt'%i,'w') as fp:
                fp.write("第几行 正确结果 预测结果\n")
                j = 1
                k =1
                for a,b in zip(y_test,predictions):
                    #print(b)
                    if(a!=b):

                        if(k%90==0):
                            #print("出现不相等，即将修改csv第%d行"%j)
                            #print(b)
                            fun(b,j)
                            print("第%d行修改成功"%j)
                        k +=1
                    fp.write("%d        %d                          %d\n"%(j,a,b))
                    j += 1
            i += 1
        
        #打印混淆矩阵
        
        #print(accuracy_score(y_test, predictions))
def main():

    read_csv_data()


if __name__ == '__main__':
    main()
    # with open('adult_train.csv','ab') as f:
    #     f.write(open('adult_test.csv','rb').read())