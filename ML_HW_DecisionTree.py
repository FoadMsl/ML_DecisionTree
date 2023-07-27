#=================================================
#   ML_HW_DecisionTree
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.9.16 & Spyder IDE
#=================================================

#%% Reset
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold



#%% Pre-processing the data

# Load the dataset
data = pd.read_csv("./heart_disease_uci.csv")


# Check data
data.head() # print first 5 rows of the dataset
data.tail() # print last 5 rows of the dataset
data.info() # get a quick description of the data
data.shape  # number of rows and columns in the dataset
data.isnull().sum() # checking for missing values


# Cleaning the data
# Drop columns which are filled mostly with NaN entries
for feat in data.columns:
    # if the column is mostly empty NaN values, drop it
    if data[feat].dropna().size<data[feat].size*0.5:
        del data[feat]

# Split dataset to the regions for more accurate data preprocessing
data["dataset"].value_counts()
data_Clev = data[data["dataset"] == 'Cleveland']
data_Hung = data[data["dataset"] == 'Hungary']
data_LongB = data[data["dataset"] == 'VA Long Beach']
data_Swit = data[data["dataset"] == 'Switzerland']

# Cleaning the data
""" Cleveland """
data_Clev.info() # get a quick description of the data
data_Clev.shape  # number of rows and columns in the dataset
data_Clev.isnull().sum() # checking for missing values
# Replace missing values with the most frequent value
data_Clev = data_Clev.fillna(data_Clev.mode().iloc[0])
data_Clev.isnull().sum() # checking for missing values

""" Hungary """
data_Hung.info() # get a quick description of the data
data_Hung.shape  # number of rows and columns in the dataset
data_Hung.isnull().sum() # checking for missing values
# Replace missing values with the most frequent value
data_Hung = data_Hung.fillna(data_Hung.mode().iloc[0])
data_Hung.isnull().sum() # checking for missing values

""" VA Long Beach """
data_LongB.info() # get a quick description of the data
data_LongB.shape  # number of rows and columns in the dataset
data_LongB.isnull().sum() # checking for missing values
# Replace missing values with the most frequent value
data_LongB = data_LongB.fillna(data_LongB.mode().iloc[0])
data_LongB.isnull().sum() # checking for missing values

""" Switzerland """
data_Swit.info() # get a quick description of the data
data_Swit.shape  # number of rows and columns in the dataset
data_Swit.isnull().sum() # checking for missing values
# Replace missing values with the most frequent value
data_Swit = data_Swit.fillna(data_Swit.mode().iloc[0])
data_Swit.isnull().sum() # checking for missing values

""" Merge sub-datasets again """
df = pd.concat([data_Clev, data_Hung, data_LongB, data_Swit])
df.head()
df.info() # get a quick description of the data
df.shape  # number of rows and columns in the dataset
df.isnull().sum() # checking for missing values


# Removing non-predictive features
df.drop(['id','dataset'],axis=1,inplace=True)

# Turn the problem to binary classification
df['num']=np.where(df['num']>0,1,0) # (condition, x, y) | Return elements chosen from x or y depending on condition.

# Converting all values into numerical values
le = LabelEncoder()
df.sex = le.fit_transform(df.sex)
df.cp = le.fit_transform(df.cp)
df.fbs = le.fit_transform(df.fbs)
df.restecg = le.fit_transform(df.restecg)
df.exang = le.fit_transform(df.exang)
df.slope = le.fit_transform(df.slope)


############################
correlation = df.corr()  # Calculate the correlation matrix of the data
plt.figure(figsize=(18, 12))  # Set the size of the figure for the heatmap
plot = sns.heatmap(correlation, cmap='jet', annot=True) ; # Create a heatmap with correlation values and annotations
plot.set_title("Correlation map")

df.skew() # Check the skewness of the data

# checking the distribution of Target Variable
df['num'].value_counts()

# Model standartization
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df['num']=np.where(df['num']>0,1,0) # (condition, x, y) | Return elements chosen from x or y depending on condition.
############################


# Check data again
df.head() # print first 5 rows of the dataset
df.tail() # print last 5 rows of the dataset
df.info() # get a quick description of the data
df.shape  # number of rows and columns in the dataset
df.isnull().sum() # checking for missing values
df.describe() # shows a summary of the numerical attributes



#%% Split the data into features and target
X = df.iloc[:,:-1] # features
y = df.iloc[:,-1] # target


#%% Model
""" Build a decision tree with the ID3 algorithm """

# From Library
# model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth='None', max_leaf_nodes='None', ccp_alpha='0.0')

# From Scratch



#%% 1 - (a) & (b)
""" Randomly select the values of 45%, 55%, 65%, and 75% of the training data 
and train the tree with it. Then test it on the entire test data and finally 
report the accuracy of the classification on the training and test data along 
with the size of the tree. Repeat this process of randomly dividing the data 
for the training process 3 times and determine the accuracy values each time, 
and finally get the average accuracy."""

# Define the fractions of the training data to use
train_sizes = [0.45, 0.55, 0.65, 0.75]

# Initialize lists to store the average accuracy and size for each fraction
avg_acc = []
avg_tree_size = []

# Initialize lists to store the accuracy and size for all repetition
accuracies = []
tree_sizes = []

# Loop over each fraction
for size in train_sizes:
    # Initialize lists to store the accuracy and size for each repetition
    acc = []
    tree_size = []
    
    # Repeat the process of randomly dividing the data for the training process three times
    for i in range(3):
        # Split the data into train and test sets | Reset each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=size)
        
        # Build & Fit a decision tree classifier
        model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best')
        # model_DT = DecisionTreeClassifier()
        model_DT.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = model_DT.predict(X_val)
        
        # Calculate the classification accuracy on validation set
        accuracy = accuracy_score(y_val, y_pred)
        
        # Append the results to the lists
        acc.append(accuracy)
        tree_size.append(model_DT.tree_.node_count)
        
        accuracies.append(accuracy)
        tree_sizes.append(model_DT.tree_.node_count)

    # Calculate the average accuracy and size for each fraction
    avg_acc_frac = sum(acc)/len(acc)
    avg_size_frac = sum(tree_size)/len(tree_size)

    # Append the results to the lists
    avg_acc.append(avg_acc_frac)
    avg_tree_size.append(avg_size_frac)


""" use the entire training data and and train the tree with it. Then test it 
on the entire test data and finally report the accuracy of the classification 
on the training and test data along with the size of the tree and determine 
the accuracy values. In the report, discuss what effect the amount of 
training data had on the accuracy of the classification on the test data and 
the size of the decision tree."""

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Build & Fit a decision tree classifier
model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best')

# model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)

# Make predictions on Test set
y_pred = model_DT.predict(X_test)

# Calculate the classification accuracy on Test set
test_accuracy = accuracy_score(y_test, y_pred)

# Append the results to the lists
accuracies.append(test_accuracy)
tree_sizes.append(model_DT.tree_.node_count)
avg_acc.append(test_accuracy)
avg_tree_size.append(model_DT.tree_.node_count)


# Print the results
print("Test Accuracies for All Fractions and Loops:", accuracies)
print("Tree Sizes for All Fractions and Loops:", tree_sizes)
print(f'Average accuracy of Each Fraction: {avg_acc}')
print(f'Average tree size of Each Fractio: {avg_tree_size}')



#%% 2 - (a)
""" Randomly select 75% of the training data as training data and the 
remaining 25% as validation data. Then build a decision tree with the help of 
training data and perform pruning with validation data. Draw the loss curve 
of classification on all 3 sets of validation, train and test for different 
number of tree nodes."""

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, random_state=42)

# Initialize lists to store the accuracy and loss for different number of tree nodes
train_acc = []
test_acc = []
val_acc = []
train_loss = []
test_loss = []
val_loss = []
nodes = []

# Loop over different number of tree nodes from 2 to 10
for node in range(2,11):
    # Build & Fit a decision tree classifier with the given number of nodes
    model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_leaf_nodes=node)
    model_DT.fit(X_train, y_train)

    # Make predictions on train, test, and validation sets
    y_pred_train = model_DT.predict(X_train)
    y_pred_test = model_DT.predict(X_test)
    y_pred_val = model_DT.predict(X_val)

    # Calculate the classification accuracy on train, test, and validation sets
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_val = accuracy_score(y_val, y_pred_val)

    # Calculate the classification loss on train, test, and validation sets
    loss_train = -acc_train
    loss_test = -acc_test
    loss_val = -acc_val

    # Append the results to the lists
    train_acc.append(acc_train)
    test_acc.append(acc_test)
    val_acc.append(acc_val)
    train_loss.append(loss_train)
    test_loss.append(loss_test)
    val_loss.append(loss_val)
    nodes.append(node)

# Plot the loss curve of classification on all three sets for different number of tree nodes
plt.plot(nodes, train_loss, label='Train')
plt.plot(nodes, test_loss, label='Test')
plt.plot(nodes, val_loss,label='Validation')
plt.xlabel('Number of tree nodes')
plt.ylabel('Classification loss')
plt.title("loss curve for 75% of the training data")
plt.legend()
plt.show()



#%% 2 - (b)
""" Perform K-fold cross validation with K=4 and select the training and 
validation data accordingly, then build the decision tree with the help of 
the training data and perform pruning with the validation data. Finally, the 
average result on all 4 systems is considered. Draw the classification loss 
curve on all 3 validation, train and test sets for different number of tree 
nodes in each state and also the average of 4 states for training and test 
sets."""

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Perform K-fold cross validation with K=4 and select the training and validation data accordingly
kf = KFold(n_splits=4)
kf.get_n_splits(X_train)

# Initialize lists to store the average accuracy and loss for different number of tree nodes
avg_train_acc = []
avg_test_acc = []
avg_val_acc = []
avg_train_loss = []
avg_test_loss = []
avg_val_loss = []
nodes = []

# Loop over different number of tree nodes from 2 to 10
for node in range(2,11):
    # Initialize lists to store the accuracy and loss for each fold
    train_acc = []
    test_acc = []
    val_acc = []
    train_loss = []
    test_loss = []
    val_loss = []

    # Loop over each fold
    for train_index, val_index in kf.split(X_train):
        # Split the data into train and validation sets
        X_train_n   = X_train.iloc[train_index]
        X_val       = X_train.iloc[val_index]
        y_train_n   = y_train.iloc[train_index]
        y_val       = y_train.iloc[val_index]

        # Build & Fit a decision tree classifier with the given number of nodes
        model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_leaf_nodes=node)
        model_DT.fit(X_train_n, y_train_n)

        # Make predictions on train and test sets
        y_pred_train = model_DT.predict(X_train_n)
        y_pred_test = model_DT.predict(X_test)
        y_pred_val = model_DT.predict(X_val)

        # Calculate the classification accuracy on train and test sets
        acc_train = accuracy_score(y_train_n, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_val = accuracy_score(y_val, y_pred_val)

        # Calculate the classification loss on train and test sets
        loss_train = -acc_train
        loss_test = -acc_test
        loss_val = -acc_val

        # Append the results to the lists
        train_acc.append(acc_train)
        test_acc.append(acc_test)
        val_acc.append(acc_val)
        train_loss.append(loss_train)
        test_loss.append(loss_test)
        val_loss.append(loss_val)

    # Calculate the average accuracy and loss for each number of nodes
    avg_train_acc_nodes = sum(train_acc)/len(train_acc)
    avg_test_acc_nodes = sum(test_acc)/len(test_acc)
    avg_val_acc_nodes = sum(val_acc)/len(val_acc)
    avg_train_loss_nodes = sum(train_loss)/len(train_loss)
    avg_test_loss_nodes = sum(test_loss)/len(test_loss)
    avg_val_loss_nodes = sum(val_loss)/len(val_loss)

    # Append the results to the lists
    avg_train_acc.append(avg_train_acc_nodes)
    avg_test_acc.append(avg_test_acc_nodes)
    avg_val_acc.append(avg_val_acc_nodes)
    avg_train_loss.append(avg_train_loss_nodes)
    avg_test_loss.append(avg_test_loss_nodes)
    avg_val_loss.append(avg_val_loss_nodes)
    nodes.append(node)

# Plot the loss curve of classification on all three sets for different number of tree nodes
plt.plot(nodes, avg_train_loss, label='Train')
plt.plot(nodes, avg_test_loss, label='Test')
plt.plot(nodes, avg_val_loss, label='Validation')
plt.xlabel('Number of tree nodes')
plt.ylabel('Classification loss')
plt.title("loss curve for K-fold cv with K=4")
plt.legend()
plt.show()