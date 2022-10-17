#Delinquency is a major metric in assessing risk as more and more customers
# getting delinquent means the risk of customers that will default will also increase.
#The main objective is to minimize the risk for which you need to build a decision tree model using CART technique
# that will identify various risk and non-risk attributes of borrower’s to get into delinquent stage
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Reading data
df = pd.read_csv("Loan Delinquent Dataset.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.columns)
df = df.drop(["ID", "delinquent"], axis = 1)
print(df)
print('term \n', df.term.value_counts(), '\n')
print('gender \n', df.gender.value_counts(), '\n')
print('purpose \n', df.purpose.value_counts(), '\n')
print('home_ownership \n', df.home_ownership.value_counts(), '\n')
print('age \n', df.age.value_counts(), '\n')
print('FICO \n', df.FICO.value_counts(), '\n')


df['purpose'] = np.where(df["purpose"] == "other", "Other", df["purpose"])
# One-Hot encoding (0 , 1)
df["term"] = np.where(df["term"] == "36 months", 0, 1)
df["gender"] = np.where(df["gender"] == "Male", 1, 0)
df["age"] = np.where(df["age"] == "20-25", 1, 0)
df["FICO"] = np.where(df["FICO"] == "300-500", 0, 1)

# Creating new columns with the name own_house, mortgage_house, Rent_house
df["own_house"] = np.where(df["home_ownership"] == "Own", 1, 0)
df["Mortgage_house"] = np.where(df["home_ownership"] == "Mortgage", 1, 0)
df["Rent_house"] = np.where(df["home_ownership"] == "Rent", 1, 0)

df["home_loan"] = np.where(df["purpose"] == "House", 1, 0)
df["Car_loan"] = np.where(df["purpose"] == "Car", 1, 0)
df["Personal_loan"] = np.where(df["purpose"] == "Personal", 1, 0)
df["Wedding_loan"] = np.where(df["purpose"] == "Wedding", 1, 0)
df["Medical_loan"] = np.where(df["purpose"] == "Medical", 1, 0)

df = df.drop(["purpose", "home_ownership"], axis=1)
print(df)

# checking for the proportion of 0 and 1 to check from class imbalance
print(df.Sdelinquent.value_counts(normalize=True))
#1    0.668601
#0    0.331399
# This shows that the class is imbalance with biasness towards 1s

# EDA

# splitting data into train and test
X = df.drop("Sdelinquent", axis=1) # selecting everything but Sdelinquent as X
Y = df.pop("Sdelinquent") # selecting Sdelinquent as Y

print(X.head)
print(Y)

# train and test split

X_train, X_test, train_labels, test_labels = train_test_split(X, Y, test_size=0.3 , random_state=425, stratify=Y)
print('Train_labels', train_labels.shape)
print('Test_labels', test_labels.shape)

X.drop([])
#Buliding the Decision Tree

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, train_labels)

from sklearn import tree

train_char_label = ['No', 'Yes']
df_Tree_File = open('df_Tree_File.dot','w')
dot_data = tree.export_graphviz(dt_model,
                                out_file=df_Tree_File,
                                feature_names = list(X_train),
                                class_names = list(train_char_label))

df_Tree_File.close()
# copy paste the content from the file generated in graphiz to see the decision tree

#Feature Engineering
print(pd.DataFrame(dt_model.feature_importances_*100, columns=["Imp"], index=X_train.columns).sort_values('Imp', ascending=False))
X.drop(['own_house', 'Mortgage_house','Rent_house', 'home_loan', 'Car_loan', 'Personal_loan', 'Wedding_loan','Medical_loan'], axis=1, inplace=True)
# dropping the columns because in feature importance they are shown as least important
print(X.columns)

Y_predict = dt_model.predict(X_test)
print(dt_model.score(X_test, test_labels))
print(dt_model.score(X_train, train_labels))

# Pruning of Decision Tree

reg_dt_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state=425)
reg_dt_model.fit(X_train, train_labels)

df_tree_regularized = open('df_tree_regularized.dot','w')
dot_data = tree.export_graphviz(reg_dt_model, out_file= df_tree_regularized , feature_names = list(X_train), class_names = list(train_char_label))
df_tree_regularized.close()
dot_data

print(reg_dt_model.score(X_train, train_labels))
print(reg_dt_model.score(X_test, test_labels))

y_train_predict=reg_dt_model.predict(X_train)
y_test_predict=reg_dt_model.predict(X_test)
print(y_test_predict)

#Getting the probability instead of just 0 and 1
y_test_predict_prob=reg_dt_model.predict_proba(X_test)
print(y_test_predict_prob)
df_prob=pd.DataFrame(y_test_predict_prob)
print(df_prob.head())

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
print(confusion_matrix(train_labels,y_train_predict))
print(classification_report(train_labels,y_train_predict))
plot_confusion_matrix(reg_dt_model,X_train,train_labels)
plt.title("Confusion Matrics Train_data")
plt.savefig("Confusion_Matrics_train_data.jpg")
plt.show()

#ROC and AUC curve

# predict probabilities
probs = reg_dt_model.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title("ROC Curve Train_Data")
plt.savefig("ROC_Curve_Train_Data")
# show the plot
plt.show()

probs_t = reg_dt_model.predict_proba(X_test)
probs_t = probs_t[:, 1]
auc = roc_auc_score(test_labels, probs_t)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(test_labels, probs_t)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.title("ROC Curve Test_Data")
plt.savefig("ROC_Curve_Test_Data")
# show the plot
plt.show()


#Confusion Metrics wrt to test_data

print(confusion_matrix(test_labels,y_test_predict))
print(classification_report(test_labels,y_test_predict))
plot_confusion_matrix(reg_dt_model,X_test,test_labels)
plt.title("Confusion Matrics Test_data")
plt.savefig("Confusion_Matrics_test_data.jpg")
plt.show()

