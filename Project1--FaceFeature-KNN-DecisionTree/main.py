'''
Author: Waiyat Hamdani
Project assigment 1 Artificial inteligence and machine learning

Note: The comment is for Dr. Housain , for future me, for future interviewer for the job. It also to having fun while write the code
    can't stay coding without sense of humor ^.^. have to keep thing interesting.
    this comment also to explain how the way I think to solve this code
'''
import os
import sklearn.datasets as datasets
import glob #pip3 install glob3 / https://docs.python.org/3/library/glob.html
import numpy as np
import pandas as pd #pip3 install pandas /sudo apt-get install python3-pandas
from sklearn import metrics #for confusion ,precision , recall ,accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

filespts=sorted(glob.glob('FaceDatabase/**/*.pts')) #glob sorting by size so that is why i sorted

alldata= []

filename=[]
for i in filespts:
    base=os.path.basename(i)
    filename.append(base)
    #print(i)
    f=open(i, "r")
    alldatas=f.readlines()[3:25] #start at 3 line because the first 3 line is just version and n_point is un-nessasary
    #print(alldatas)
    alldata.append(alldatas)
    alldatalen=len(alldatas)

    #rewriting the point line into float
    for xy in range(alldatalen):
        alldatas[xy]= alldatas[xy].rstrip().split()
        alldatas[xy][0] = float(alldatas[xy][0]) #this is for changing x if dunno read back about list ^.^
        alldatas[xy][1] = float(alldatas[xy][1]) #this is for changing y


def printdatapoint():
    data = {}
    no=0
    for i in filename:
        data.update({i:alldata[no]})
        no+=1
    df = pd.DataFrame(data)
    print(df)


def definingfeature():
    print("------------------first defining feature:eye length ratio---------------")
    global alldataextraction
    alldataextraction=[]
    dataeyelength=[]


    no=0
    for i in range(40):
        point9x=alldata[i][9][0]               #if you dont know about this . because my array result is [[[]]] so the first one is for the position of 40 faces
        point9y=alldata[i][9][1]               # since one faces having 22 pts of xy that one is for the middle
        point10x=alldata[i][10][0]             # so of course now you undestand the last list for. if you got confuses is for [xy]
        point10y=alldata[i][10][1]
        point11x=alldata[i][11][0]
        point11y=alldata[i][11][1]
        point12x=alldata[i][12][0]
        point12y=alldata[i][12][1]
        point8x=alldata[i][8][0]
        point8y=alldata[i][8][1]
        point13x=alldata[i][13][0]
        point13y=alldata[i][13][1]
        distance9to10=(((point9x-point10x)**(2))+((point9y-point10y)**(2)))**(1/2)
        distance11to12=(((point11x-point12x)**(2))+((point11y-point12y)**(2)))**(1/2)
        distance8to13=(((point8x-point13x)**(2))+((point8y-point13y)**(2)))**(1/2)
        if distance9to10 > distance11to12:
            result=distance9to10/distance8to13
            #print(result)
            dataeyelength.append(result)
                #print(result)
        else:
            result=distance11to12/distance8to13#print(result)str(result[no])3
            #print(result)
            dataeyelength.append(result)
        no+=1
    df2 = pd.DataFrame(dataeyelength ,index=filename,columns=['values'])
    print(df2)

    print("------------------second defining feature:eye distance---------------")

    eyedistanceratio=[]

    no=0
    for i in range(40):
        point0x=alldata[i][0][0]
        point0y=alldata[i][0][1]
        point1x=alldata[i][1][0]
        point1y=alldata[i][1][1]
        point8x=alldata[i][8][0]
        point8y=alldata[i][8][1]
        point13x=alldata[i][13][0]
        point13y=alldata[i][13][1]
        distance0to1=(((point0x-point1x)**(2))+((point0y-point1y)**(2)))**(1/2)
        distance8to13=(((point8x-point13x)**(2))+((point8y-point13y)**(2)))**(1/2)
        result=distance0to1/distance8to13
        eyedistanceratio.append(result)

        no+=1
    df3 = pd.DataFrame(eyedistanceratio,index=filename,columns=['values'])
    print(df3)

    print("------------------third defining feature:nose ratio---------------")

    noseratio=[]

    no=0
    for i in range(40):
        point15x=alldata[i][15][0]
        point15y=alldata[i][15][1]
        point16x=alldata[i][16][0]
        point16y=alldata[i][16][1]
        point20x=alldata[i][20][0]
        point20y=alldata[i][20][1]
        point21x=alldata[i][21][0]
        point21y=alldata[i][21][1]
        distance15to16=(((point15x-point16x)**(2))+((point15y-point16y)**(2)))**(1/2)
        distance20to21=(((point20x-point21x)**(2))+((point20y-point21y)**(2)))**(1/2)
        result=distance15to16/distance20to21
        noseratio.append(result)

        no+=1
    df4 = pd.DataFrame(noseratio,index=filename,columns=['values'])
    print(df4)

    print("------------------fourth defining feature:lips size ratio---------------")

    lipSratio=[]

    no=0
    for i in range(40):
        point2x=alldata[i][2][0]
        point2y=alldata[i][2][1]
        point3x=alldata[i][3][0]
        point3y=alldata[i][3][1]
        point17x=alldata[i][17][0]
        point17y=alldata[i][17][1]
        point18x=alldata[i][18][0]
        point18y=alldata[i][18][1]
        distance2to3=(((point2x-point3x)**(2))+((point2y-point3y)**(2)))**(1/2)
        distance17to18=(((point17x-point18x)**(2))+((point17y-point18y)**(2)))**(1/2)
        result=distance2to3/distance17to18
        lipSratio.append(result)

        no+=1
    df5 = pd.DataFrame(lipSratio,index=filename,columns=['values'])
    print(df5)

    print("------------------fifth defining feature:Lip length ratio---------------")

    lipLratio=[]

    no=0
    for i in range(40):
        point2x=alldata[i][2][0]
        point2y=alldata[i][2][1]
        point3x=alldata[i][3][0]
        point3y=alldata[i][3][1]
        point20x=alldata[i][20][0]
        point20y=alldata[i][20][1]
        point21x=alldata[i][21][0]
        point21y=alldata[i][21][1]
        distance2to3=(((point2x-point3x)**(2))+((point2y-point3y)**(2)))**(1/2)
        distance20to21=(((point20x-point21x)**(2))+((point20y-point21y)**(2)))**(1/2)
        result=distance2to3/distance20to21
        lipLratio.append(result)

        no+=1
    df6 = pd.DataFrame(lipLratio,index=filename,columns=['values'])
    print(df6)

    print("------------------Six defining feature:Eye-brow length ratio---------------")

    EyebrowLratio=[]

    no=0
    for i in range(40):
        point4x=alldata[i][4][0]
        point4y=alldata[i][4][1]
        point5x=alldata[i][5][0]
        point5y=alldata[i][5][1]
        point6x=alldata[i][6][0]
        point6y=alldata[i][6][1]
        point7x=alldata[i][7][0]
        point7y=alldata[i][7][1]
        point8x=alldata[i][8][0]
        point8y=alldata[i][8][1]
        point13x=alldata[i][13][0]
        point13y=alldata[i][13][1]
        distance4to5=(((point4x-point5x)**(2))+((point4y-point5y)**(2)))**(1/2)
        distance6to7=(((point6x-point7x)**(2))+((point6y-point7y)**(2)))**(1/2)
        distance8to13=(((point8x-point13x)**(2))+((point8y-point13y)**(2)))**(1/2)
        if distance4to5 > distance6to7:
            result=distance4to5/distance8to13
            #print(result)
            EyebrowLratio.append(result)
                #print(result)
        else:
            result=distance6to7/distance8to13#print(result)str(result[no])3
            #print(result)
            EyebrowLratio.append(result)


        no+=1
    df7 = pd.DataFrame(EyebrowLratio,index=filename,columns=['values'])
    print(df7)

    print("------------------seventh defining feature:aggresive ratio---------------")

    aggresiveratio=[]

    no=0
    for i in range(40):
        point10x=alldata[i][10][0]
        point10y=alldata[i][10][1]
        point19x=alldata[i][19][0]
        point19y=alldata[i][19][1]
        point20x=alldata[i][20][0]
        point20y=alldata[i][20][1]
        point21x=alldata[i][21][0]
        point21y=alldata[i][21][1]
        distance10to19=(((point10x-point19x)**(2))+((point10y-point19y)**(2)))**(1/2)
        distance20to21=(((point20x-point21x)**(2))+((point20y-point21y)**(2)))**(1/2)
        result=distance10to19/distance20to21
        aggresiveratio.append(result)

        no+=1
    df8 = pd.DataFrame(aggresiveratio,index=filename,columns=['values'])
    print(df8)

    for i in range (40):
        alldataextraction.append([dataeyelength[i],eyedistanceratio[i],noseratio[i],lipSratio[i],lipLratio[i],EyebrowLratio[i],aggresiveratio[i]])





def ResultsAndAnalysis():
    trainData = []
    trainTarget = []

    testData = []
    testClasses = []
    #class 1
    for i in range(0,3):                #for those of you don't understand to count 0,3 is taking position 0,1,2
        trainData.append(alldataextraction[i])
        trainTarget.append(1)               #this is for class
    testData.append(alldataextraction[3])   #load one testData
    testClasses.append(1)

    #class 2
    for i in range(4,7):
        trainData.append(alldataextraction[i])    #for those of you don't understand to count 0,3 is taking position 4,5,6
        trainTarget.append(2)              #this is for class
    testData.append(alldataextraction[7])   #load one testData
    testClasses.append(2)

    #class 3
    for i in range(8,11):               #for those of you don't understand to count 0,3 is taking position 8,9,10
        trainData.append(alldataextraction[i])
        trainTarget.append(3)           #this is for class
    testData.append(alldataextraction[11])  #load one testData
    testClasses.append(3)

    #class 5
    for i in range(12,15):              #for those of you don't understand how to count 0,3 is taking position 12,13,14
        trainData.append(alldataextraction[i])
        trainTarget.append(4)           #this is for class
    testData.append(alldataextraction[15])  #load one testData
    testClasses.append(4)

    #class 5
    for i in range(16,19):              #for those of you don't understand how to count 0,3 is taking position 16,17,18
        trainData.append(alldataextraction[i])
        trainTarget.append(5)           #this is for class
    testData.append(alldataextraction[19])  #load one testData
    testClasses.append(5)

    #class 6
    for i in range(20,23):              #for those of you don't understand how to count 0,3 is taking position 20,21,22
        trainData.append(alldataextraction[i])
        trainTarget.append(6)           #this is for class
    testData.append(alldataextraction[23])  #load one testData
    testClasses.append(6)

    #class 7
    for i in range(24,27):              #for those of you don't understand how to count 0,3 is taking position 24,25,26
        trainData.append(alldataextraction[i])
        trainTarget.append(7)           #this is for class
    testData.append(alldataextraction[27])  #load one testData
    testClasses.append(7)

    #class 8
    for i in range(28,31):                 #for those of you don't understand how to count 0,3 is taking position 28,29,30
        trainData.append(alldataextraction[i])
        trainTarget.append(8)                #this is for class
    testData.append(alldataextraction[31])  #load one testData
    testClasses.append(8)

    #class 9
    for i in range(32,35):                  #for those of you don't understand how to count 0,3 is taking position 32,33,34
        trainData.append(alldataextraction[i])
        trainTarget.append(9)               #this is for class
    testData.append(alldataextraction[35])  #load one testData
    testClasses.append(9)

    #class 10
    for i in range(36,39):                  #for those of you don't understand how to count 0,3 is taking position 36,37,38
        trainData.append(alldataextraction[i])
        trainTarget.append(10)              #this is for class
    testData.append(alldataextraction[39])  #load one testData
    testClasses.append(10)


    trainData=np.asarray(trainData)

    print("\n my train data length:{}".format(len(trainData)))
    trainTarget=np.asarray(trainTarget)
    testData=np.asarray(testData)
    print("\n my test data length:{}".format(len(testData)))

    #owh the bottom comment just for the question. don't get confused ^.^
    '''
    In this assignment, you are going to identify persons from face images using K-Nearest Neighbors and Decision Tree classifier.
    1 You will have to prepare a confusion matrix and calculate precision, recall rate, and accuracy for each of the classifiers.
    '''

    #calculating knn
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    classifier=KNeighborsClassifier(n_neighbors=2)
    classifier.fit(trainData,trainTarget)
    decisions=classifier.predict(testData)              #since i have the decision I can load them for analysis


    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    confMetricReport=metrics.confusion_matrix(testClasses, decisions)
    '''
    df9=pd.DataFrame(confMetricReport)          TO CONFUSING USING DataFrame SO IM JUST GOING TO PRINT THEM
    print(df9)
    '''
    print("\n confusion matric report knn: \n {}".format(confMetricReport))
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    PrecisionS=metrics.precision_score(testClasses, decisions, average='micro')
    #ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
    # i choose macro first they giving me this
    # UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
    print("\n precision matric report micro knn: \n {}".format(PrecisionS))
    recalrate=metrics.recall_score(testClasses,decisions, average='micro')
    print("\n recal rate matric report micro knn: \n {}".format(recalrate))
    accuracys=metrics.accuracy_score(testClasses,decisions)
    print("\n accuracy matric report micro knn: \n {}".format(accuracys))

    # calculating DecisionTree
    #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    entropy=DecisionTreeClassifier(criterion="entropy")
    entropy.fit(trainData,trainTarget)
    decisionstree=entropy.predict(testData)
    #print(decisionstree)
    confMetricReporttree=metrics.confusion_matrix(testClasses, decisionstree)
    print("\n confusion matric report decision tree : \n {}".format(confMetricReporttree))
    PrecisionStree=metrics.precision_score(testClasses, decisionstree, average='micro' )
    print("\n precision matric report micro tree: \n {}".format(PrecisionStree))
    recalratetree=metrics.recall_score(testClasses,decisionstree, average='micro')
    print("\n recal rate matric report micro tree: \n {}".format(recalratetree))
    accuracystree=metrics.accuracy_score(testClasses,decisionstree)
    print("\n accuracy matric report micro tree: \n {}".format(accuracystree))


if __name__ == "__main__":
    printdatapoint()
    definingfeature()
    ResultsAndAnalysis()
