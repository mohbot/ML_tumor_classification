%reset
import pandas as pd


tcga_path = ("path_to_input_files")
cl_df = pd.read_csv(tcga_path+'tcgaClinical_df.csv')


print("\n\n shape of clinical list: {}".format(cl_df.shape))
print("\n\n head of clinical list")
print(cl_df.head(1))

print("\n\n sample_type breadown")
print(cl_df.sample_type.value_counts())
print("\n\n select only primary tumor")
cl_df = cl_df.loc[cl_df.sample_type=='primary.solid.tumor']

print("\n\n creating cancer_types and cleaning")
print("\n\n projects: {}".format(cl_df.project.unique()))
cl_df = cl_df.loc[~cl_df.project.isnull()]
cl_df = cl_df.reset_index(drop=True)
cl_df['cancer_type'] = cl_df.project.apply(lambda x: x.split('-')[1])

print("\n\n loading expression data")
tpm_df = pd.read_pickle(tcga_path+'tcgaTpmLog2_df.pkl')
print("\n\n head of expression data")
print(tpm_df[tpm_df.columns[:3]].head(1))
tpm_df = tpm_df.set_index('Hugo_Symbol')

print("\n\n select right intersecting samples of cl and exp. data")
barcodes = set(cl_df.barcode).intersection(set(tpm_df.columns))
tpm_df = tpm_df[barcodes]


print("\n\n number of patients")
print(cl_df.loc[cl_df.barcode.isin(barcodes),'patient'].shape)
print("\n\n number of unique patients")
print(cl_df.loc[cl_df.barcode.isin(barcodes),'patient'].unique().shape)

cl_df = cl_df.loc[cl_df.barcode.isin(barcodes)]
cl_df = cl_df.reset_index(drop=True)

print("\n\n cancer_types: {}".format(cl_df.cancer_type.unique().size))
print("\n\n cancer_types: {}".format(cl_df.cancer_type.unique().tolist()))
print("\n\n tail of cancer_types breakdown: \n{}".format(
cl_df.cancer_type.value_counts().tail(3)))

print("since smallest cancer set has at least 30 samples we select all cancer types")


################################################################################
# for pca and t-sne 

#### scaler ####
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import string
from sklearn.manifold import TSNE


n_comps = 1000
perplexity = 50
n_iter = 2000
early_exagger = 150


tpm_df = tpm_df.T

# Standardizing the features
tpm = StandardScaler().fit_transform(tpm_df.values)
#### end of scaler ####


pca = PCA(n_components=n_comps)
pca.fit(tpm)
tpm_pca = pca.transform(tpm)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
ax = plt.gca()
ax.set_facecolor('#DCDCDC')
tpm_pcs_fn = 'tpm_pcs_cumSum_{}.pdf'.format(n_comps)

print('output file name: {}'.format(tpm_pcs_fn))
plt.savefig(tpm_pcs_fn)
plt.close()

################################################################################
# for creating a GBM method
#### scaler ####
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle


# reduce the number of genes to 1382
# gnImpIntDf = pd.read_csv('gnImpIntDf.csv')
# tpm_df = tpm_df[gnImpIntDf.geneIds.values[:genCutOff_n]]


n_thread = 6

tpm_df = tpm_df.T


tpm_dis_df = tpm_df.merge(cl_df[['barcode','cancer_type']],
                          left_on=tpm_df.index, 
                          right_on='barcode',how='inner')


tpm_dis_df = tpm_dis_df.set_index('barcode',drop=True)


le = preprocessing.LabelEncoder()
le.fit(tpm_dis_df.cancer_type)

print('start of classification')
# stratified k-fold cross validation evaluation of xgboost model
# split data into X and y
X = tpm_dis_df.loc[:,tpm_dis_df.columns !='cancer_type']
Y = le.transform(tpm_dis_df.cancer_type.values)


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y,
                                                    test_size=0.3, 
                                                    random_state=10)

# fit model no training data
model = xgb.XGBClassifier(nthread=n_thread)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("test Accuracy: %.2f%%" % (accuracy * 100.0))

# test Accuracy: 90.58%

confusion_matrix_testDf = pd.DataFrame(columns=list(le.inverse_transform(range(le.classes_.size))),
                                       index=list(le.inverse_transform(range(le.classes_.size))),
                                       data = confusion_matrix(y_test, y_pred))

pickle.dump(model, open("tpm_disModelRndSt7_{}.pickle.dat".format('all'), "wb"))

confusion_matrix_testDf.to_csv('confusion_matrix_test_{}_Df.csv'.format('all'),index=False)

geneImportance_df = pd.DataFrame(columns=['GeneId','Importance'],
                            data=zip(genes,list(model.feature_importances_)))

geneImportance_df = geneImportance_df.sort_values(by='Importance',ascending=False)

geneImportance_df.to_csv('gene_importance_{}_Df.csv'.format('all'),index=False)


y_pred = model.predict(X_train)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_train, predictions)
print("train Accuracy: %.2f%%" % (accuracy * 100.0))


################################################################################
# find the number of non-zero genes
# for creating a GBM method
#### scaler ####
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle

genes = tpm_df.columns.values
tpm_df = tpm_df.T
gnImpIntDf = pd.read_csv('gene_importance_all_df.csv')

genCutOff_ns = [3000,2000,1000,500,200,100,50]
accur_train = []
accur_test = []

for genCutOff_n in genCutOff_ns:
    print(genCutOff_n)
    # reduce the number of genes to ~3K
    tpm_df = tpm_df[gnImpIntDf.GeneId.values[:genCutOff_n]]

    tpm_dis_df = tpm_df.merge(cl_df[['barcode','cancer_type']],
                          left_on=tpm_df.index, 
                          right_on='barcode',how='inner')

    tpm_dis_df = tpm_dis_df.set_index('barcode',drop=True)


    le = preprocessing.LabelEncoder()
    le.fit(tpm_dis_df.cancer_type)

    print('start of classification')
    # stratified k-fold cross validation evaluation of xgboost model
    # split data into X and y
    X = tpm_dis_df.loc[:,tpm_dis_df.columns !='cancer_type']
    Y = le.transform(tpm_dis_df.cancer_type.values)


    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y,
                                                        test_size=0.3,
                                                        random_state=10)

    # fit model no training data
    model = xgb.XGBClassifier(nthread=6)
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("test Accuracy: %.2f%%" % (accuracy * 100.0))
    accur_test.append(accuracy)

    # test Accuracy: 90.58%

    confusion_matrix_testDf = pd.DataFrame(columns=list(le.inverse_transform(range(le.classes_.size))),
                                           index=list(le.inverse_transform(range(le.classes_.size))),
                                           data = confusion_matrix(y_test, y_pred))

    pickle.dump(model, open("tpm_disModelRndSt7_{}.pkl".format(genCutOff_n), "wb"))

    confusion_matrix_testDf.to_csv('confusion_matrix_test_{}_Df.csv'.format(genCutOff_n),index=False)

    geneImportance_df = pd.DataFrame(columns=['GeneId','Importance'],
                                data=zip(genes,list(model.feature_importances_)))

    geneImportance_df = geneImportance_df.sort_values(by='Importance',ascending=False)

    geneImportance_df.to_csv('gene_importance_{}_Df.csv'.format(genCutOff_n),index=False)


    y_pred = model.predict(X_train)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_train, predictions)
    print("train Accuracy: %.2f%%" % (accuracy * 100.0))
    accur_train.append(accuracy)
    
    
traTstDf = pd.DataFrame(columns=['genes_n','train','test'])

traTstDf.genes_n = genCutOff_ns
traTstDf.train = accur_train
traTstDf.test = accur_test

plt.plot(traTstDf.genes_n,traTstDf.train,'-o')
plt.plot(traTstDf.genes_n,traTstDf.test,'-o')
plt.ylim([0,105])
plt.xlabel("number of genes")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("feature reduction")
plt.savefig("feature_reduction.pdf")
plt.show()

################################################################################
# plot heatmap of 200 genes
import pandas as pd
from matplotlib.pyplot import cm
import seaborn as sns

gnImpDf = pd.read_csv('gene_importance_200_Df.csv')

genes = tpm_df.index.values

tpm_df = tpm_df.T

tpm_df = tpm_df[gnImpDf.GeneId]

tpm_df['barcode'] = tpm_df.index

tpm_df = tpm_df.merge(cl_df[['barcode','cancer_type']])

tpm_df = tpm_df.drop(columns=['barcode'])

DiseaseType = tpm_df.pop('cancer_type')

colors=cm.nipy_spectral(pd.np.linspace(0,1,DiseaseType.unique().size))
lut = dict(zip(DiseaseType.unique(), colors))

row_colors = DiseaseType.map(lut)
fig = sns.clustermap(tpm_df, row_colors=row_colors,
                     yticklabels=False,xticklabels=1)

fig.ax_heatmap.set_xticklabels(fig.ax_heatmap.get_xmajorticklabels(), 
                                fontsize = 2.5)

fig.ax_row_dendrogram.set_visible(False)
fig.ax_col_dendrogram.set_visible(False)

fig.cax.set_visible(False)

import matplotlib.patches as mpatches
legend_dis = [mpatches.Patch(color=lut[dis], label=dis) for dis in lut.keys()]

fig.ax_heatmap.legend(loc='upper left',
                      prop={'size': 6},
                      bbox_to_anchor=(1, 1),
                      handles=legend_dis)

fig.savefig("disease_heatmap.pdf",bbox_inches='tight')
