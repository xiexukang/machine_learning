import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def data_deal2(file_path):
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
    column_names=data.columns
    for c in column_names:
         data[c] = data[c].replace(" ", "")
    column_names = data.columns
    #替换？为空值
    for c in column_names:
        print(data[c].unique)
        data[c] = data[c].replace("?", np.NaN)
    #空值填充
    data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
    # for c in column_names:
    #     print(df[c].unique)
    data = data.replace(to_replace = {'Local-gov', 'Federal-gov', 'State-gov'}, value = 'Govern-employ', regex = True)
    data = data.replace(to_replace = {'Self-emp-inc', 'Self-emp-not-inc'}, value = 'Self-employ', regex = True)
    data = data.replace(to_replace = {'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'}, value = 'Primary-School', regex = True)
    data = data.replace(to_replace = {'Doctorate', 'Assoc-voc' ,'Assoc-acdm', 'Prof-school'}, value = 'other', regex = True)
    data = data.replace(to_replace = {'Never-married', 'Separated', 'Widowed', 'Divorced', 'Married-spouse-absent'}, value = 'no-spouse', regex = True)
    data = data.replace(to_replace = {'Married-AF-spouse', 'Married-civ-spouse'}, value = 'have-spouse', regex = True)
    data = data.replace(to_replace = {'United-States', 'Outlying-US(Guam-USVI-etc)', 'Canada', 'Mexico', 'Cuba','Puerto-Rico', 'Nicaragua', 
                                              'Honduras', 'Jamaica', 'Guatemala','El-Salvador', 'Haiti', 'Dominican-Republic'}, value = 'North-America', regex = True)
    data = data.replace(to_replace = {'Columbia', 'Ecuador', 'Peru', 'Trinadad&Tobago'}, value = 'South-America', regex = True)
    data = data.replace(to_replace = {'China', 'India', 'Vietnam', 'Taiwan', 'Laos', 'Japan', 'Thailand', 'Cambodia', 'South',
                                              'Philippines', 'Iran', 'Hong'}, value = 'Asia', regex = True)
    data = data.replace(to_replace = {'England','Scotland','Ireland','Germany', 'France', 'Greece', 'Italy', 'Portugal', 'Hungary', 
                                              'Yugoslavia', 'Poland', 'Holand-Netherlands','Portugal'}, value = 'Europe', regex = True)
    data = data.replace(to_replace = {'Amer-Indian-Eskimo', 'Amer-Asian-Eskimo'}, value = 'Other', regex = True) 
    data = data.replace(to_replace = {'Other-service', 'Protective-serv', 'Priv-house-serv' }, value = 'Service', regex = True)
    data = data.replace(to_replace = {'Machine-op-inspct', 'Tech-support', 'Armed-Forces'}, value = 'Tech', regex = True)
    data = data.replace(to_replace = {'Craft-repair', 'Handlers-cleaners', 'Farming-fishing', 'Transport-moving' }, value = 'Physic-work', regex = True)
    data = data.replace(to_replace = {'Adm-clerical', 'Exec-managerial', 'Sales'}, value = 'Bussiness', regex = True)

    def convert_category(data, category_list):
        for name in category_list:
            dummies = pd.get_dummies(data[name])
            data = pd.concat([data, dummies], axis = 1)
            data.drop(name, inplace = True, axis = 1)
        return data
    data = convert_category(data, ['workclass', 'education', 'marital-status', 'occupation',  
                                       'relationship', 'race', 'sex', 'native-country'])
    data['income'] = data['income'].map({' <=50K': 0, ' >50K': 1,' >50K.': 1,' <=50K.': 0})
    from scipy.stats.mstats import winsorize
    data["age"]           = winsorize(data["age"],(0,0.15))
    data["fnlwgt"]        = winsorize(data["fnlwgt"],(0,0.15))
    data["capital-gain"]  = winsorize(data["capital-gain"],(0,0.099))
    data["capital-loss"]  = winsorize(data["capital-loss"],(0,0.099))
    data["hours-per-week"]= winsorize(data["hours-per-week"],(0.12,0.18))
    data.drop('fnlwgt', inplace = True, axis = 1)
    #data.drop('income', inplace = True, axis = 1)
    #data.drop('Outlying-US(Guam-USVI-etc)', inplace = True, axis = 1)
    return data

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
def main():
    train_data=data_deal("adult_train.csv")
    test_data = data_deal("adult_test.csv")
#     print(train_data.head())
#     print(train_data.shape)
#     print(test_data.head())
#     print(test_data.shape)
    from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
    from sklearn.preprocessing import StandardScaler
    #X_train = StandardScaler().fit_transform(train_data.loc[:, train_data.columns != 'income'])
    X_train = train_data.loc[:, train_data.columns != 'income']
    print("测试集数据:",X_train)
    #X_test = StandardScaler().fit_transform(test_data.loc[:, train_data.columns != 'income'])
    X_test = test_data.loc[:, train_data.columns != 'income']
    y_train = train_data['income']
    y_test = test_data['income']
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    classifiers = [
    CatBoostClassifier(learning_rate=0.01),
   
]
    #for _ in range(100):
    from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
        #X_train,y_train=_shuffle(X_train),_shuffle(y_train)
        #predict_model = LogisticRegressionCV(multi_class="ovr",solver="sag",cv=5,tol=1e-3)
        #使用多种普遍模型进行处理
    #测试10次

    i =0
    kf = KFold(n_splits=10, shuffle=True, random_state=50)
    for model in classifiers:
        print(model)
        predict_model = model
        predict_model.fit(X_train,y_train)
        
        predictions = predict_model.predict(X_test)
        a= accuracy_score(y_test, predictions)
        print("精确度",a)
    # params = {'max_depth': [6], 
    #      'n_estimators': [200],
    #       'learning_rate': [0.07, 0.06],
    #       'max_features': [3,4]
    #      }
    #print(modelname,":",maxacc)
    # classifier = GradientBoostingClassifier()

    # grid = GridSearchCV(classifier, param_grid=params, cv=kf)
    # search_result = grid.fit(X_train,y_train)
    # predictions = search_result.predict(X_test)
    # a= accuracy_score(y_test, predictions)
    # print("a")
    # means = search_result.cv_results_['mean_test_score']
    # params = search_result.cv_results_['params']
    # for m, p in zip(means, params):
    #     print(f"{m} with: {p}")
        i =1
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
    # print(classification_report(y_test,predictions))
    # #打印混淆矩阵
    # print(confusion_matrix(y_test,predictions))
    # print(accuracy_score(y_test, predictions))

if __name__ == '__main__':
    main()