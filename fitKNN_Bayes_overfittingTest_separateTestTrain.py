from numpy import ndarray,argsort, arange,std
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
import matplotlib as mpl
# from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from ds_charts import get_variable_types, choose_grid, HEIGHT, bar_chart, multiple_line_chart
import ds_charts as ds
from sklearn.metrics import accuracy_score,recall_score,f1_score
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
import cupy as cp
# from cuml.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from matplotlib.pyplot import imread, imshow, axis
from subprocess import call
from pathlib import Path
import time
import cuml
from torch import nn
# from cuml.model_selection import train_test_split
import sys
import cudf 
import sklearn.model_selection
import os
from sklearn.neural_network import MLPClassifier
from joblib import dump,load
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from xgboost import XGBClassifier
def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    ds.multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'images/overfitting_{name}.png')










#############################################################
# SÓ MUDAR ISTO

filetag = f'../datasets/health/health_train_smote.csv'
filetagTest = f'../datasets/health/health_test.csv'
separator = ","
target = 'readmitted'

##########################################################3


if len(sys.argv) >= 4:
    filetag = str(sys.argv[1])
    filetagTest = str(sys.argv[2])
    target = str(sys.argv[3])

if len(sys.argv) >=5:
    separator = str(sys.argv[4])


print(f"Filetag: {filetag}")
print(f"Filetag for test file: {filetagTest}")
print(f"Target: {target}")
print(f"Separator: {separator}")
iter = 0

data = read_csv(filetagTest,sep=",")
classes = cudf.DataFrame(list(map(str, data.pop(target).values.tolist())))
has_negatives = False
try:
    if "date" in data.columns:
        data.pop('date')
except:
    pass
vars = cudf.DataFrame(data.values)
if (data.values < 0).any():
    has_negatives = True
    print("\n\n HAS NEGATIVES \n \n")
idx_extra = 1
if has_negatives:
    idx_extra = 0

tstX = cp.array(data.values,cp.float32)
tstY = cp.array(classes.to_numpy(),cp.float32).T[0]













data = read_csv(filetag,sep=",")
classes = cudf.DataFrame(list(map(str, data.pop(target).values.tolist())))
has_negatives = False
try:
    if "date" in data.columns:
        data.pop('date')
except:
    pass
vars = cudf.DataFrame(data.values)
if (data.values < 0).any():
    has_negatives = True
    print("\n\n HAS NEGATIVES \n \n")
idx_extra = 1
if has_negatives:
    idx_extra = 0

trnX = cp.array(data.values,cp.float32)
trnY = cp.array(classes.to_numpy(),cp.float32).T[0]









# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   All algorithms assume tstX, tstY, trnX and trnY are passed as prepared above.
#
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






















# PREPARE IMAGE FILE DIR AND FOLDER


figSaveFormat = "pdf"
subDirs = filetag.split(sep="/")
filename = subDirs[-1]
filename_no_ext = filename.split(sep=".")[0]
imageDir =  ""
modelDir = ""


subDirs.remove(filename)
for sd in subDirs:
    imageDir += f"{sd}/"
    modelDir += f"{sd}/"
imageDir += f"images/"
modelDir += f"models/"

try:
    os.mkdir(imageDir)
except:
    print("image dir already exists")
try:
    os.mkdir(modelDir)
except:
    print("model dir already exists")
imageDir += f"{filename_no_ext}/"
modelDir += f"{filename_no_ext}/"
try:
    os.mkdir(imageDir)
except:
    print("file image dir already exists")
try:
    os.mkdir(modelDir)
except:
    print("file model dir already exists")

# ---------------------------------------------------------------------------------------------------------------------------
#
#                       NB
#
#----------------------------------------------------------------------------------------------------------------------------
doNB = False

if doNB:
    fig, axs = mpl.pyplot.subplots(2+idx_extra, 3, figsize=(3*HEIGHT, (2+idx_extra)*HEIGHT), squeeze=False)

    print(trnY.get())
    labels = unique(trnY.get())
    labels.sort()


    prd_trn_by_model = {}
    prd_tst_by_model = {}

    graph_pos = 0


    print("GaussianNB")
    clf = GaussianNB()
    clf.fit(trnX,trnY )
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    prd_trn_by_model["GaussianNB"] = prd_trn
    prd_tst_by_model["GaussianNB"] = prd_tst
    ds.plot_evaluation_results(labels, trnY.get(), prd_trn.get(), tstY.get(), prd_tst.get(),axs[graph_pos,0],axs[graph_pos,1],axs[graph_pos,2],"Gaussian NB")
    graph_pos+=1



    print("BernoulliNB")
    clf = BernoulliNB()
    clf.fit(trnX,trnY )
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    prd_trn_by_model["BernoulliNB"] = prd_trn
    prd_tst_by_model["BernoulliNB"] = prd_tst
    ds.plot_evaluation_results(labels, trnY.get(), prd_trn.get(), tstY.get(), prd_tst.get(),axs[graph_pos,0],axs[graph_pos,1],axs[graph_pos,2],"Bernoulli NB")
    graph_pos+=1


    if not has_negatives:
        print("MultinomialNB")
        clf = MultinomialNB()
        clf.fit(trnX,trnY )
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        prd_trn_by_model["MultinomialNB"] = prd_trn
        prd_tst_by_model["MultinomialNB"] = prd_tst
        ds.plot_evaluation_results(labels, trnY.get(), prd_trn.get(), tstY.get(), prd_tst.get(),axs[graph_pos,0],axs[graph_pos,1],axs[graph_pos,2],"Multinomial NB")
        graph_pos+=1

    savefig(f'{imageDir}NB_results.png')
    fig, axs = mpl.pyplot.subplots(1, 3, figsize=(3*HEIGHT, 1*HEIGHT), squeeze=False)
    graph_pos = 0


    xvalues = []
    yvalues = []
    for clf in prd_tst_by_model.keys():
        xvalues.append(str(clf))
        yvalues.append(accuracy_score(tstY.get(), prd_tst_by_model[clf].get()))
    bar_chart(xvalues, yvalues,ax=axs[0,0], title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)

    xvalues = []
    yvalues = []
    for clf in prd_tst_by_model.keys():
        xvalues.append(str(clf))
        yvalues.append(recall_score(tstY.get(), prd_tst_by_model[clf].get(),average="micro"))
    bar_chart(xvalues, yvalues,ax=axs[0,1], title='Comparison of Naive Bayes Models', ylabel='recall', percentage=True)


    xvalues = []
    yvalues = []
    for clf in prd_tst_by_model.keys():
        xvalues.append(str(clf))
        yvalues.append(f1_score(tstY.get(), prd_tst_by_model[clf].get(),average="micro"))
    bar_chart(xvalues, yvalues,ax=axs[0,2], title='Comparison of Naive Bayes Models', ylabel='f1-score', percentage=True)

    savefig(f'{imageDir}NB_Variants.{figSaveFormat}', format=figSaveFormat)



















# ---------------------------------------------------------------------------------------------------------------------------
#
#                       KNN
#
#----------------------------------------------------------------------------------------------------------------------------
doKNN = False
doBestValsKNN = True
doOverfittingKNN = True

if doKNN:
    prd_trn_by_model = {}
    prd_tst_by_model = {}
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19,25,31,39,45,51,61,71,81,91,101]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    # nvalues = [1, 3]
    # dist = ['manhattan', 'euclidean','chebyshev']
    values = {}
    for d in dist:
        for n in nvalues:
            print(f"KNN_{d}_{n}")
            clf = KNeighborsClassifier(n_neighbors=n, metric=d)
            clf.fit(trnX, trnY)
            prd_trn = clf.predict(trnX)
            prd_tst = clf.predict(tstX)
            prd_trn_by_model[f"KNN_{d}_{n}"] = prd_trn
            prd_tst_by_model[f"KNN_{d}_{n}"] = prd_tst

    accuracies = {}
    recalls = {}
    f1_scores = {}

    for d in dist:
        accs = []
        recs = []
        f1s = []
        for n in nvalues:
            accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"KNN_{d}_{n}"].get()))
            recs.append(recall_score(tstY.get(), prd_tst_by_model[f"KNN_{d}_{n}"].get(),average="micro"))
            f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"KNN_{d}_{n}"].get(),average="micro"))
        accuracies[d] = accs
        recalls[d] = recs
        f1_scores[d] = f1s


    fig, axs = mpl.pyplot.subplots(1, 3, figsize=(3*HEIGHT, 1*HEIGHT), squeeze=False)

    multiple_line_chart(nvalues, accuracies,ax=axs[0,0], title='Accuracy score by n', xlabel='n', ylabel="accuracy", percentage=True)
    multiple_line_chart(nvalues, recalls,ax=axs[0,1], title='Recall score by n', xlabel='n', ylabel="recall", percentage=True)
    multiple_line_chart(nvalues, f1_scores,ax=axs[0,2], title='F1-Score by n', xlabel='n', ylabel="f1-score", percentage=True)
    savefig(f'{imageDir}KNN_VARIANTS_iter_{iter}_CUDA.{figSaveFormat}', format=figSaveFormat)





















    # ------------------------
    #
    #   Overfitting tests KNN
    #
    #-------------------------
    if doOverfittingKNN:
        accuraciesT = {}
        recallsT = {}
        f1_scoresT = {}
        for d in dist:
            accs = []
            recs = []
            f1s = []
            for n in nvalues:
                accs.append(accuracy_score(trnY.get(),prd_trn_by_model[f"KNN_{d}_{n}"].get()))
                recs.append(recall_score(trnY.get(), prd_trn_by_model[f"KNN_{d}_{n}"].get(),average="micro"))
                f1s.append(f1_score(trnY.get(), prd_trn_by_model[f"KNN_{d}_{n}"].get(),average="micro"))
            accuraciesT[d] = accs
            recallsT[d] = recs
            f1_scoresT[d] = f1s


        fig, axs = mpl.pyplot.subplots(3, 3, figsize=(3*HEIGHT, 3*HEIGHT), squeeze=False)

        multiple_line_chart(nvalues, {"train":accuraciesT[dist[0]],"test":accuracies[dist[0]]},ax=axs[0,0], title=f"Overfitting KNN_{dist[0]}", xlabel='n', ylabel="accuracy", percentage=True)
        multiple_line_chart(nvalues, {"train":recallsT[dist[0]],"test":recalls[dist[0]]},ax=axs[0,1], title=f"Overfitting KNN_{dist[0]}", xlabel='n', ylabel="recall", percentage=True)
        multiple_line_chart(nvalues, {"train":f1_scoresT[dist[0]],"test":f1_scores[dist[0]]},ax=axs[0,2], title=f"Overfitting KNN_{dist[0]}", xlabel='n', ylabel="f1-score", percentage=True)

        multiple_line_chart(nvalues, {"train":accuraciesT[dist[1]],"test":accuracies[dist[1]]},ax=axs[1,0], title=f"Overfitting KNN_{dist[1]}", xlabel='n', ylabel="accuracy", percentage=True)
        multiple_line_chart(nvalues, {"train":recallsT[dist[1]],"test":recalls[dist[1]]},ax=axs[1,1], title=f"Overfitting KNN_{dist[1]}", xlabel='n', ylabel="recall", percentage=True)
        multiple_line_chart(nvalues, {"train":f1_scoresT[dist[1]],"test":f1_scores[dist[1]]},ax=axs[1,2], title=f"Overfitting KNN_{dist[1]}", xlabel='n', ylabel="f1-score", percentage=True)

        multiple_line_chart(nvalues, {"train":accuraciesT[dist[2]],"test":accuracies[dist[2]]},ax=axs[2,0], title=f"Overfitting KNN_{dist[2]}", xlabel='n', ylabel="accuracy", percentage=True)
        multiple_line_chart(nvalues, {"train":recallsT[dist[2]],"test":recalls[dist[2]]},ax=axs[2,1], title=f"Overfitting KNN_{dist[2]}", xlabel='n', ylabel="recall", percentage=True)
        multiple_line_chart(nvalues, {"train":f1_scoresT[dist[2]],"test":f1_scores[dist[2]]},ax=axs[2,2], title=f"Overfitting KNN_{dist[2]}", xlabel='n', ylabel="f1-score", percentage=True)
        savefig(f'{imageDir}KNN_VARIANTS_OVERFITTING.{figSaveFormat}', format=figSaveFormat)

    # ------------------------
    #
    #   Best result Values KNN
    #
    #-------------------------

    if doBestValsKNN:


        for d_n in range(len(dist)):
            for nval in range(len(nvalues)):
                graph_pos = d_n * len(nvalues) + nval
                d = dist[d_n]
                n = nvalues[nval]
                print(f"calculating evaluation results of: KNN_{d}_{n}")
                fig, axs = mpl.pyplot.subplots(1, 3, figsize=(3*HEIGHT, 1*HEIGHT), squeeze=False)
                ds.plot_evaluation_results(labels, trnY.get(), prd_trn_by_model[f"KNN_{d}_{n}"].get(), tstY.get(), prd_tst_by_model[f"KNN_{d}_{n}"].get(),axs[0,0],axs[0,1],axs[0,2],f"KNN {d} {n}")
                savefig(f'{imageDir}KNN_EVAL_RESULTS_{d}_{n}.{figSaveFormat}', format=figSaveFormat)




















# ---------------------------------------------------------------------------------------------------------------------------
#
#                       DECISION TREES
#
#----------------------------------------------------------------------------------------------------------------------------
doDecisionTrees = True
doExportDecisionTreeGraph = False
doGraphsDT = True
doOverfittingDT= True
doFeatureImportanceDT = False
printFeatureImportanceDT = False
doBestValsDT = True


if doDecisionTrees:
    prd_trn_by_model = {}
    prd_tst_by_model = {}

    # min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    # max_depths = [2, 5, 10, 15, 20, 25]

    # min_impurity_decrease = [0.01, 0.005]
    # max_depths = [2,3]
    
    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005,0.0001,0]
    max_depths = [2,5, 10, 15, 20, 25,35,55]



    # should be contained within max_depths, otherwise blank plots will show up
    max_depths_for_impurityGraph = [x for x in [2,5, 10, 15, 20, 25,35,55] if x in max_depths ]

    criteria = ['entropy', 'gini']
    
    importancesByModel = {}

    for c in criteria:
        for d in max_depths:
            for imp in min_impurity_decrease:
                print(f"DT_{c}_{d}_{imp}")
                clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=imp)
                clf.fit(trnX.get(), trnY.get())
                prd_trn = clf.predict(trnX.get())
                prd_tst = clf.predict(tstX.get())
                prd_trn_by_model[f"DT_{c}_{d}_{imp}"] = prd_trn
                prd_tst_by_model[f"DT_{c}_{d}_{imp}"] = prd_tst


                if doExportDecisionTreeGraph:
                    dot_data = export_graphviz(clf, out_file=f'{imageDir}DT_{c}_{d}_{imp}.dot', filled=True, rounded=True, special_characters=True)
                    call(['dot', '-Tpdf', f'{imageDir}DT_{c}_{d}_{imp}.dot', '-o', f'{imageDir}DT_{c}_{d}_{imp}.pdf'])
                    tmp_path = Path(f'{imageDir}DT_{c}_{d}_{imp}.dot')
                    tmp_path.unlink(missing_ok=True)
                
                if doFeatureImportanceDT:
                    importancesByModel[f"DT_{c}_{d}_{imp}"] = clf.feature_importances_

    
    if doGraphsDT:

        fig, axs = mpl.pyplot.subplots(2, 3, figsize=(3*HEIGHT, 2*HEIGHT), squeeze=False)
        accuracies = {}
        recalls = {}
        f1_scores = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d in max_depths_for_impurityGraph:
                accs = []
                recs = []
                f1s = []
                for imp in min_impurity_decrease:
                    accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"DT_{c}_{d}_{imp}"]))
                    recs.append(recall_score(tstY.get(), prd_tst_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                    f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                accuracies[f"{d}"] = accs
                recalls[f"{d}"] = recs
                f1_scores[f"{d}"] = f1s
            multiple_line_chart(min_impurity_decrease, accuracies,ax=axs[c_n,0], title=f'Decision Tree with {c} criteria', xlabel='min_impurity_decrease', ylabel="accuracy", percentage=True)
            multiple_line_chart(min_impurity_decrease, recalls,ax=axs[c_n,1],title=f'Decision Tree with {c} criteria', xlabel='min_impurity_decrease', ylabel="recall", percentage=True)
            multiple_line_chart(min_impurity_decrease, f1_scores,ax=axs[c_n,2],title=f'Decision Tree with {c} criteria', xlabel='min_impurity_decrease', ylabel="f1-score", percentage=True)
            
        savefig(f'{imageDir}DT_Results_By_Criteria.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}DT_Results_By_Criteria.png")

    if doOverfittingDT:
        # overfiting graph
        
        accuracies = {}
        recalls = {}
        f1_scores = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for imp in min_impurity_decrease:
                accs = []
                recs = []
                f1s = []
                for d in max_depths:
                    accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"DT_{c}_{d}_{imp}"]))
                    recs.append(recall_score(tstY.get(), prd_tst_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                    f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                accuracies[f"{c}_{imp}"] = accs
                recalls[f"{c}_{imp}"] = recs
                f1_scores[f"{c}_{imp}"] = f1s

        
        accuraciesT = {}
        recallsT = {}
        f1_scoresT = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for imp in min_impurity_decrease:
                accs = []
                recs = []
                f1s = []
                for d in max_depths:
                    accs.append(accuracy_score(trnY.get(),prd_trn_by_model[f"DT_{c}_{d}_{imp}"]))
                    recs.append(recall_score(trnY.get(), prd_trn_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                    f1s.append(f1_score(trnY.get(), prd_trn_by_model[f"DT_{c}_{d}_{imp}"],average="micro"))
                accuraciesT[f"{c}_{imp}"] = accs
                recallsT[f"{c}_{imp}"] = recs
                f1_scoresT[f"{c}_{imp}"] = f1s
        
        fig, axs = mpl.pyplot.subplots(len(min_impurity_decrease)*len(criteria), 3, figsize=(3*HEIGHT, len(min_impurity_decrease)*len(criteria)*HEIGHT), squeeze=False)
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            len_imp = len(min_impurity_decrease)
            for imp_n in range(len_imp):
                imp = min_impurity_decrease[imp_n]
                name = f"{c}_{imp}"
                multiple_line_chart(max_depths, {"train":accuraciesT[name],"test":accuracies[name]},ax=axs[c_n*len_imp+imp_n,0], title=f"Overfitting DT {name}_imp", xlabel='max depth', ylabel="accuracy", percentage=True)
                multiple_line_chart(max_depths, {"train":recallsT[name],"test":recalls[name]},ax=axs[c_n*len_imp+imp_n,1], title=f"Overfitting DT {name}_imp", xlabel='max depth', ylabel="recall", percentage=True)
                multiple_line_chart(max_depths, {"train":f1_scoresT[name],"test":f1_scores[name]},ax=axs[c_n*len_imp+imp_n,2], title=f"Overfitting DT {name}_imp", xlabel='max depth', ylabel="f1-score", percentage=True)
        savefig(f'{imageDir}DT_Overfitting_Results.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}DT_Overfitting_Results.png")


    # Feature importance graph
    if doFeatureImportanceDT:
        variables = data.columns
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for imp_n in range(len(min_impurity_decrease)):
                imp = min_impurity_decrease[imp_n]
                for d_n in range(len(max_depths)):
                    d=max_depths[d_n]
                    print(f"Feature importances for " + f"DT_{c}_{d}_{imp}")
                    importances = importancesByModel[f"DT_{c}_{d}_{imp}"]
                    indices = argsort(importances)[::-1]
                    elems = []
                    imp_values = []
                    for f in range(len(variables)):
                        elems += [variables[indices[f]]]
                        imp_values += [importances[indices[f]]]
                        
                        if printFeatureImportanceDT:
                            print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')
                            
                    fig, axs = mpl.pyplot.subplots(1, 1, figsize=(2*HEIGHT, 2*HEIGHT), squeeze=False)
                    ds.horizontal_bar_chart(elems, imp_values, ax=axs[0,0],error=None, title=f"DT_{c}_{d}_{imp}"+' Features importance', xlabel='importance', ylabel='variables')
                    savefig(f'{imageDir}DT_Feature_Importance_{c}_{d}_{imp}.{figSaveFormat}', format=figSaveFormat)


    if doBestValsDT:
        labels = unique(trnY.get())
        labels.sort()
        for c in criteria:
            for d in max_depths:
                for imp in min_impurity_decrease:
                    print(f"calculating evaluation results of: DT_{c}_{d}_{imp}")
                    fig, axs = mpl.pyplot.subplots(1, 3, figsize=(5*HEIGHT, 1*HEIGHT), squeeze=False)
                    ds.plot_evaluation_results(labels, trnY.get(), prd_trn_by_model[f"DT_{c}_{d}_{imp}"], tstY.get(), prd_tst_by_model[f"DT_{c}_{d}_{imp}"],axs[0,0],axs[0,1],axs[0,2],f"DT {c} depth:{d} imp:{imp}")
                    savefig(f'{imageDir}DT_EVAL_RESULTS_{c}_{d}_{imp}.{figSaveFormat}', format=figSaveFormat)




















# ---------------------------------------------------------------------------------------------------------------------------
#
#                       RANDOM FOREST
#
#----------------------------------------------------------------------------------------------------------------------------
doRandomForest = False
doGraphsRF = True
doOverfittingRF= True
doFeatureImportanceRF = False
printFeatureImportanceRF = False
doBestValsRF = False

if doRandomForest:
    criteria = ['entropy']
    # criteria = ['entropy', 'gini']
    
    n_estimators = [1,2,3,4,5, 10, 25, 50, 75, 100, 200, 300, 400]
    max_depths = [5, 10,25]
    max_features = [.3, .5, .7, 1]
    # n_estimators = [200]
    # max_depths = [25]
    # max_features = [.7]
    importancesByModel = {}
    stdevsByModel = {}
    prd_trn_by_model = {}
    prd_tst_by_model = {}


    for c in criteria:
        for d in max_depths:
            for f in max_features:
                for estim in n_estimators:
                    if doFeatureImportanceRF:
                        ti = time.process_time()
                        print(f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}")
                        clf = sklearn.ensemble.RandomForestClassifier(max_depth=d, criterion=c, max_features=f,n_estimators=estim)
                        clf.fit(trnX.get(), trnY.get())
                        prd_trn = clf.predict(trnX.get())
                        prd_tst = clf.predict(tstX.get())
                        tf = time.process_time()
                        delta = tf-ti
                        print(f"Took sklearn: {delta} seconds")
                        prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = prd_trn
                        prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = prd_tst
                    else:
                        ti = time.process_time()
                        print(f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}")
                        clf = cuml.ensemble.RandomForestClassifier(max_depth=d, max_features=f,n_estimators=estim)
                        clf.fit(trnX, trnY)
                        prd_trn = clf.predict(trnX)
                        prd_tst = clf.predict(tstX)
                        tf = time.process_time()
                        delta = tf-ti
                        print(f"Took: {delta} seconds")
                        prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = prd_trn.get()
                        prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = prd_tst.get()
                    
                    if doFeatureImportanceRF:
                        importancesByModel[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = clf.feature_importances_
                        stdevsByModel[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"] = std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    
    
    if doGraphsRF:

        lmd= len(max_depths)
        lc=len(criteria)
        l_mf=len(max_features)
        fig, axs = mpl.pyplot.subplots(lmd*lc, 3, figsize=(3*HEIGHT, lmd*lc*HEIGHT), squeeze=False)
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                accuracies = {}
                recalls = {}
                f1_scores = {}
                for f_n in range(len(max_features)):
                    f = max_features[f_n]
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"]))
                        recs.append(recall_score(tstY.get(), prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                    print(f"Accs for {c} {d} {f}: {accs}")
                    accuracies[f"{f}"] = accs
                    recalls[f"{f}"] = recs
                    f1_scores[f"{f}"] = f1s
                multiple_line_chart(n_estimators, accuracies,ax=axs[c_n*lmd+d_n,0], title=f'Random Forest {c} depth:{d} ', xlabel='n_estimators', ylabel="accuracy", percentage=True)
                multiple_line_chart(n_estimators, recalls,ax=axs[c_n*lmd+d_n,1],title=f'Random Forest {c} dep:{d}  ', xlabel='n_estimators', ylabel="recall", percentage=True)
                multiple_line_chart(n_estimators, f1_scores,ax=axs[c_n*lmd+d_n,2],title=f'Random Forest {c} dep:{d}  ', xlabel='n_estimators', ylabel="f1-score", percentage=True)
                
        savefig(f'{imageDir}RF_Results_By_Criteria.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}RF_Results_By_Criteria.png")



    if doOverfittingRF:
        # overfiting graph
        accuracies = {}
        recalls = {}
        f1_scores = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f in max_features:
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"]))
                        recs.append(recall_score(tstY.get(), prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                    accuracies[f"RF_{c}_depth-{d}_max_features-{f}"] = accs
                    recalls[f"RF_{c}_depth-{d}_max_features-{f}"] = recs
                    f1_scores[f"RF_{c}_depth-{d}_max_features-{f}"] = f1s

        
        accuraciesT = {}
        recallsT = {}
        f1_scoresT = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f in max_features:
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(trnY.get(),prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"]))
                        recs.append(recall_score(trnY.get(), prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(trnY.get(), prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],average="micro"))
                    accuraciesT[f"RF_{c}_depth-{d}_max_features-{f}"] = accs
                    recallsT[f"RF_{c}_depth-{d}_max_features-{f}"] = recs
                    f1_scoresT[f"RF_{c}_depth-{d}_max_features-{f}"] = f1s
        
        
        lmd= len(max_depths)
        lc=len(criteria)
        l_mf=len(max_features)
        fig, axs = mpl.pyplot.subplots(lmd*lc*l_mf, 3, figsize=(3*HEIGHT, lmd*lc*l_mf*HEIGHT), squeeze=False)

        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f_n in range(len(max_features)):
                    f = max_features[f_n]
                    name = f"RF_{c}_depth-{d}_max_features-{f}"
                    multiple_line_chart(n_estimators, {"train":accuraciesT[name],"test":accuracies[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n,0], title=f"Overfitting RF_{c}_depth-{d}_max_features-{f}", xlabel='n_estimators', ylabel="accuracy", percentage=True)
                    multiple_line_chart(n_estimators, {"train":recallsT[name],"test":recalls[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n,1], title=f"Overfitting RF_{c}_depth-{d}_max_features-{f}", xlabel='n_estimators', ylabel="recall", percentage=True)
                    multiple_line_chart(n_estimators, {"train":f1_scoresT[name],"test":f1_scores[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n ,2], title=f"Overfitting RF_{c}_depth-{d}_max_features-{f}", xlabel='n_estimators', ylabel="f1-score", percentage=True)
        savefig(f'{imageDir}RF_Overfitting_Results.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}RF_Overfitting_Results.png")


    # Feature importance graph
    if doFeatureImportanceRF:
        variables = data.columns

        for c in criteria:
            for d in max_depths:
                for f in max_features:
                    for estim in n_estimators:
                        print(f"Feature importances for " + f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}")
                        name = f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"
                        importances = importancesByModel[name]
                        stdevs = stdevsByModel[name]
                        indices = argsort(importances)[::-1]
                        elems = []
                        for v in range(len(variables)):
                            elems += [variables[indices[v]]]
                            if printFeatureImportanceRF:
                                print(f'{v+1}. feature {elems[v]} ({importances[indices[v]]})')
                                
                        fig, axs = mpl.pyplot.subplots(1, 1, figsize=(2*HEIGHT, 2*HEIGHT), squeeze=False)
                        ds.horizontal_bar_chart(elems, importances[indices], stdevs[indices],ax=axs[0,0],title=f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"+' Features importance', xlabel='importance', ylabel='variables')
                        savefig(f'{imageDir}RF_Feature_Importance_{c}_depth-{d}_max_features-{f}_estimators-{estim}.{figSaveFormat}', format=figSaveFormat)


    if doBestValsRF:
        labels = unique(trnY.get())
        labels.sort()
        for c in criteria:
            for d in max_depths:
                for f in max_features:
                    for estim in n_estimators:
                        print(f"calculating evaluation results of: RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}")
                        fig, axs = mpl.pyplot.subplots(1, 3, figsize=(5*HEIGHT, 1*HEIGHT), squeeze=False)
                        ds.plot_evaluation_results(labels, trnY.get(), prd_trn_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"], tstY.get(), prd_tst_by_model[f"RF_{c}_depth-{d}_max_features-{f}_estimators-{estim}"],axs[0,0],axs[0,1],axs[0,2],f"RF_{c} depth-{d} max_features-{f} estimators-{estim}")
                        savefig(f'{imageDir}RF_EVAL_RESULTS_{c}_depth-{d}_max_features-{f}_estimators-{estim}.{figSaveFormat}', format=figSaveFormat)


















# ---------------------------------------------------------------------------------------------------------------------------
#
#                       MULTILAYER PERCEPTRON
#
#----------------------------------------------------------------------------------------------------------------------------
doMLP = False
doGraphsMLP = False
doOverfittingMLP = False
doBestValsMLP = False
doLossCurveMLP = True

if doMLP:

    print("tstX shape: ",tstX.shape)
    print("tstY shape: ",tstY.shape, " ",np.max(tstY)+1)

    lr_type = ['constant', 'invscaling', 'adaptive']
    max_iter = [100, 300, 500, 750, 1000, 2500, 5000,10000]
    learning_rate = [.001,0.01,0.1,1]

    print("n classes:",np.max(tstY.get())+1)
    print(f"MLP")


    prd_trn_by_model = {}
    prd_tst_by_model = {}

    # Generate and train network for a config, then save it to a file
    for k in range(len(lr_type)):
        d = lr_type[k]
        for lr in learning_rate:
            for n in max_iter:
                print(f"MLP_dist-{d}_lr-{lr}_max-iter-{n}")
                if os.path.exists(f"{modelDir}MLP_dist-{d}_lr-{lr}_max-iter-{n}.joblib"):
                    clf = load(f"{modelDir}MLP_dist-{d}_lr-{lr}_max-iter-{n}.joblib")
                else:
                    clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                            learning_rate_init=lr, max_iter=n, verbose=False)
                    clf.fit(trnX.get(), trnY.get())
                    # save model test
                    if not os.path.exists(f"{modelDir}MLP_dist-{d}_lr-{lr}_max-iter-{n}.joblib"):
                        dump(clf,f"{modelDir}MLP_dist-{d}_lr-{lr}_max-iter-{n}.joblib",compress=4)
                
                prd_trn = clf.predict(trnX.get())
                prd_tst = clf.predict(tstX.get())
                prd_trn_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"] = prd_trn
                prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"] = prd_tst

                if doLossCurveMLP:
                    fig, axs = mpl.pyplot.subplots(1, 1, figsize=(1*HEIGHT, 1*HEIGHT), squeeze=False)
                    v_dict = {}
                    for i in range(len(clf.loss_curve_)):
                        v_dict[i] = clf.loss_curve_[i]
                    multiple_line_chart(range(len(clf.loss_curve_)),  {"test":clf.loss_curve_},ax=axs[0,0],title=f'MLP loss curve', xlabel='iter', ylabel="loss", percentage=True)
                    savefig(f'{imageDir}MLP_LOSS_CURVE_dist-{d}_lr-{lr}_max-iter-{n}.{figSaveFormat}', format=figSaveFormat)
    if doGraphsMLP:

        lmd= len(lr_type)
        l_mf=len(learning_rate)
        fig, axs = mpl.pyplot.subplots(lmd, 3, figsize=(3*HEIGHT, lmd*HEIGHT), squeeze=False)

        for d_n in range(len(lr_type)):
            d = lr_type[d_n]
            accuracies = {}
            recalls = {}
            f1_scores = {}
            for f_n in range(len(learning_rate)):
                lr = learning_rate[f_n]
                accs = []
                recs = []
                f1s = []
                for n in max_iter :
                    accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"]))
                    recs.append(recall_score(tstY.get(), prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                    f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                print(f"Accs for {d} {lr} {n}: {accs}")
                accuracies[f"{lr}"] = accs
                recalls[f"{lr}"] = recs
                f1_scores[f"{lr}"] = f1s
            multiple_line_chart(max_iter, accuracies,ax=axs[d_n,0], title=f'MLP dep:{d} ', xlabel='max iter', ylabel="accuracy", percentage=True)
            multiple_line_chart(max_iter, recalls,ax=axs[d_n,1],title=f'MLP dep:{d}  ', xlabel='max iter', ylabel="recall", percentage=True)
            multiple_line_chart(max_iter, f1_scores,ax=axs[d_n,2],title=f'MLP dep:{d}  ', xlabel='max iter', ylabel="f1-score", percentage=True)
            
        savefig(f'{imageDir}MLP_Results_By_Criteria.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}MLP_Results_By_Criteria.png")



    if doOverfittingMLP:
        # overfiting graph

        lmd= len(lr_type)
        l_mf=len(learning_rate)
        fig, axs = mpl.pyplot.subplots(lmd*l_mf, 3, figsize=(3*HEIGHT, lmd*l_mf*HEIGHT), squeeze=False)

        accuracies = {}
        recalls = {}
        f1_scores = {}
        for d_n in range(len(lr_type)):
            d = lr_type[d_n]
            for f_n in range(len(learning_rate)):
                lr = learning_rate[f_n]
                accs = []
                recs = []
                f1s = []
                for n in max_iter :
                    accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"]))
                    recs.append(recall_score(tstY.get(), prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                    f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                accuracies[f"MLP_dist-{d}_lr-{lr}"] = accs
                recalls[f"MLP_dist-{d}_lr-{lr}"] = recs
                f1_scores[f"MLP_dist-{d}_lr-{lr}"] = f1s

        
        accuraciesT = {}
        recallsT = {}
        f1_scoresT = {}
        for d_n in range(len(lr_type)):
            d = lr_type[d_n]
            for f_n in range(len(learning_rate)):
                lr = learning_rate[f_n]
                accs = []
                recs = []
                f1s = []
                for n in max_iter :
                    accs.append(accuracy_score(trnY.get(),prd_trn_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"]))
                    recs.append(recall_score(trnY.get(), prd_trn_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                    f1s.append(f1_score(trnY.get(), prd_trn_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],average="micro"))
                accuraciesT[f"MLP_dist-{d}_lr-{lr}"] = accs
                recallsT[f"MLP_dist-{d}_lr-{lr}"] = recs
                f1_scoresT[f"MLP_dist-{d}_lr-{lr}"] = f1s

        for d_n in range(len(lr_type)):
            d = lr_type[d_n]
            for f_n in range(len(learning_rate)):
                lr = learning_rate[f_n]
                name = f"MLP_dist-{d}_lr-{lr}"
                multiple_line_chart(max_iter, {"train":accuraciesT[name],"test":accuracies[name]},ax=axs[d_n*l_mf + f_n,0], title=f"Overfitting MLP_dist-{d}_lr-{lr}", xlabel='max iter', ylabel="accuracy", percentage=True)
                multiple_line_chart(max_iter, {"train":recallsT[name],"test":recalls[name]},ax=axs[d_n*l_mf + f_n,1], title=f"Overfitting MLP_dist-{d}_lr-{lr}", xlabel='max iter', ylabel="recall", percentage=True)
                multiple_line_chart(max_iter, {"train":f1_scoresT[name],"test":f1_scores[name]},ax=axs[d_n*l_mf + f_n ,2], title=f"Overfitting MLP_dist-{d}_lr-{lr}", xlabel='max iter', ylabel="f1-score", percentage=True)
        savefig(f'{imageDir}MLP_Overfitting_Results.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}MLP_Overfitting_Results.png")

    if doBestValsMLP:
        labels = unique(trnY.get())
        labels.sort()
        for k in range(len(lr_type)):
            d = lr_type[k]
            for lr in learning_rate:
                for n in max_iter:
                    print(f"calculating evaluation results of: MLP_dist-{d}_lr-{lr}_max-iter-{n}")
                    fig, axs = mpl.pyplot.subplots(1, 3, figsize=(5*HEIGHT, 1*HEIGHT), squeeze=False)
                    ds.plot_evaluation_results(labels, trnY.get(), prd_trn_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"], tstY.get(), prd_tst_by_model[f"MLP_dist-{d}_lr-{lr}_max-iter-{n}"],axs[0,0],axs[0,1],axs[0,2],f"MLP_dist-{d}_lr-{lr}_max-iter-{n}")
                    savefig(f'{imageDir}MLP_EVAL_RESULTS_dist-{d}_lr-{lr}_max-iter-{n}.{figSaveFormat}', format=figSaveFormat)





# ---------------------------------------------------------------------------------------------------------------------------
#
#                       GRADIENT BOOSTING
#
#----------------------------------------------------------------------------------------------------------------------------
doGradBoost = False
doGraphsGB = False
doOverfittingGB= False
doFeatureImportanceGB = True
printFeatureImportanceGB = False
doBestValsGB = False

if doGradBoost:
    print("Entering Gradient boosting")
# Gradient Boost
    # clf = XGBClassifier(n_estimators=100, max_depth=15, learning_rate=0.1,n_jobs=4,tree_method="gpu_hist",gpu_id=0,predictor="gpu_predictor",max_bin=1000)

    
    n_estimators = [5, 10, 25, 50, 75, 100, 250, 500, 1000]
    max_depths = [5, 10, 25]
    learning_rate = [0.01,.05,.1, .5, .9]

    # n_estimators = [400]
    # max_depths = [25]
    # learning_rate = [ .1]

    n_estimators = [1000,]
    max_depths = [5, ]
    learning_rate = [0.01,]

    importancesByModel = {}
    stdevsByModel = {}
    prd_trn_by_model = {}
    prd_tst_by_model = {}

    criteria = [""]
    for c in criteria:
        for d in max_depths:
            for f in learning_rate:
                for estim in n_estimators:
                    ti = time.process_time()
                    print(f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}")
                    # clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=estim, max_depth=d, learning_rate=f,max_features=0.7)
                    # clf = sklearn.ensemble.HistGradientBoostingClassifier( max_depth=d, learning_rate=f)
                    clf = XGBClassifier(n_estimators=estim, max_depth=d, learning_rate=f,n_jobs=1,tree_method="gpu_hist",gpu_id=0,predictor="gpu_predictor",max_bin=1000)
                    clf.fit(trnX.get(), trnY.get())
                    prd_trn = clf.predict(trnX.get())
                    prd_tst = clf.predict(tstX.get())
                    tf = time.process_time()
                    delta = tf-ti
                    print(f"Took sklearn: {delta} seconds")
                    prd_trn_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"] = prd_trn
                    prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"] = prd_tst

                    if doFeatureImportanceGB:
                        importancesByModel[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"] = clf.feature_importances_
    
    
    if doGraphsGB:

        lmd= len(max_depths)
        lc=len(criteria)
        l_mf=len(learning_rate)
        fig, axs = mpl.pyplot.subplots(lmd*lc, 3, figsize=(3*HEIGHT, lmd*lc*HEIGHT), squeeze=False)
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                accuracies = {}
                recalls = {}
                f1_scores = {}
                for f_n in range(len(learning_rate)):
                    f = learning_rate[f_n]
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"]))
                        recs.append(recall_score(tstY.get(), prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                    print(f"Accs for {c} {d} {f} {estim}: {accs}")
                    accuracies[f"{f}"] = accs
                    recalls[f"{f}"] = recs
                    f1_scores[f"{f}"] = f1s
                multiple_line_chart(n_estimators, accuracies,ax=axs[c_n*lmd+d_n,0], title=f'Gradient Boosting {c} depth:{d} ', xlabel='n_estimators', ylabel="accuracy", percentage=True)
                multiple_line_chart(n_estimators, recalls,ax=axs[c_n*lmd+d_n,1],title=f'Gradient Boosting {c} dep:{d}  ', xlabel='n_estimators', ylabel="recall", percentage=True)
                multiple_line_chart(n_estimators, f1_scores,ax=axs[c_n*lmd+d_n,2],title=f'Gradient Boosting {c} dep:{d}  ', xlabel='n_estimators', ylabel="f1-score", percentage=True)
                
        savefig(f'{imageDir}GB_Results_By_Criteria.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}GB_Results_By_Criteria.png")



    if doOverfittingGB:
        # overfiting graph
        accuracies = {}
        recalls = {}
        f1_scores = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f in learning_rate:
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(tstY.get(),prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"]))
                        recs.append(recall_score(tstY.get(), prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(tstY.get(), prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                    accuracies[f"GB_{c}_depth-{d}_learning_rate-{f}"] = accs
                    recalls[f"GB_{c}_depth-{d}_learning_rate-{f}"] = recs
                    f1_scores[f"GB_{c}_depth-{d}_learning_rate-{f}"] = f1s

        
        accuraciesT = {}
        recallsT = {}
        f1_scoresT = {}
        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f in learning_rate:
                    accs = []
                    recs = []
                    f1s = []
                    for estim in n_estimators :
                        accs.append(accuracy_score(trnY.get(),prd_trn_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"]))
                        recs.append(recall_score(trnY.get(), prd_trn_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                        f1s.append(f1_score(trnY.get(), prd_trn_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],average="micro"))
                    accuraciesT[f"GB_{c}_depth-{d}_learning_rate-{f}"] = accs
                    recallsT[f"GB_{c}_depth-{d}_learning_rate-{f}"] = recs
                    f1_scoresT[f"GB_{c}_depth-{d}_learning_rate-{f}"] = f1s
        
        
        lmd= len(max_depths)
        lc=len(criteria)
        l_mf=len(learning_rate)
        fig, axs = mpl.pyplot.subplots(lmd*lc*l_mf, 3, figsize=(3*HEIGHT, lmd*lc*l_mf*HEIGHT), squeeze=False)

        for c_n in range(len(criteria)):
            c = criteria[c_n]
            for d_n in range(len(max_depths)):
                d = max_depths[d_n]
                for f_n in range(len(learning_rate)):
                    f = learning_rate[f_n]
                    name = f"GB_{c}_depth-{d}_learning_rate-{f}"
                    multiple_line_chart(n_estimators, {"train":accuraciesT[name],"test":accuracies[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n,0], title=f"Overfitting GB_{c}_depth-{d}_learning_rate-{f}", xlabel='n_estimators', ylabel="accuracy", percentage=True)
                    multiple_line_chart(n_estimators, {"train":recallsT[name],"test":recalls[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n,1], title=f"Overfitting GB_{c}_depth-{d}_learning_rate-{f}", xlabel='n_estimators', ylabel="recall", percentage=True)
                    multiple_line_chart(n_estimators, {"train":f1_scoresT[name],"test":f1_scores[name]},ax=axs[c_n*lmd*l_mf+d_n*l_mf+f_n ,2], title=f"Overfitting GB_{c}_depth-{d}_learning_rate-{f}", xlabel='n_estimators', ylabel="f1-score", percentage=True)
        savefig(f'{imageDir}GB_Overfitting_Results.{figSaveFormat}', format=figSaveFormat)
        print(f"Saved {imageDir}GB_Overfitting_Results.png")


    # Feature importance graph
    if doFeatureImportanceGB:
        variables = data.columns

        for c in criteria:
            for d in max_depths:
                for f in learning_rate:
                    for estim in n_estimators:
                        print(f"Feature importances for " + f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}")
                        name = f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"
                        importances = importancesByModel[name]
                        indices = argsort(importances)[::-1]
                        elems = []
                        for v in range(len(variables)):
                            elems += [variables[indices[v]]]
                            if printFeatureImportanceRF:
                                print(f'{v+1}. feature {elems[v]} ({importances[indices[v]]})')
                                
                        fig, axs = mpl.pyplot.subplots(1, 1, figsize=(2*HEIGHT, 2*HEIGHT), squeeze=False)
                        ds.horizontal_bar_chart(elems, importances[indices],ax=axs[0,0],title=f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"+' Features importance', xlabel='importance', ylabel='variables')
                        savefig(f'{imageDir}GB_Feature_Importance_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}.{figSaveFormat}', format=figSaveFormat)


    if doBestValsGB:
        labels = unique(trnY.get())
        labels.sort()
        for c in criteria:
            for d in max_depths:
                for f in learning_rate:
                    for estim in n_estimators:
                        print(f"calculating evaluation results of: GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}")
                        fig, axs = mpl.pyplot.subplots(1, 3, figsize=(5*HEIGHT, 1*HEIGHT), squeeze=False)
                        ds.plot_evaluation_results(labels, trnY.get(), prd_trn_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"], tstY.get(), prd_tst_by_model[f"GB_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}"],axs[0,0],axs[0,1],axs[0,2],f"GB_{c} depth-{d} learning_rate-{f} estimators-{estim}")
                        savefig(f'{imageDir}GB_EVAL_RESULTS_{c}_depth-{d}_learning_rate-{f}_estimators-{estim}.{figSaveFormat}', format=figSaveFormat)

