import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc,mean_squared_error, r2_score
from BP_Class import BP_Class
from plots import Plots_Class
import warnings
 
plotsClass = Plots_Class()

warnings.filterwarnings("ignore")

# Load dataset
file_path = 'A2-ring/A2-ring-separable-normalized.csv'

label="Ring_separable"

# Load data into DataFrames
df = pd.read_csv(file_path, delimiter=',')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and test sets with a 80:20 ratio
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

bpclass = BP_Class()

bpclass.printPlots(df,label)

epochs=[50,100]

# Keras model
model1 = Sequential()
model1.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
model1.add(Dense(4, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model2 = Sequential()
model2.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model3 = Sequential()
model3.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model4 = Sequential()
model4.add(Dense(8, activation='relu', input_dim=X_train.shape[1]))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

models=[]
models.append(("model1",model1))
models.append(("model2",model2))
models.append(("model3",model3))
models.append(("model4",model4))

best_error=0
best_epoch=0
best_model_name=""
best_accuracy=0

for epoch in epochs:
    for model in models:
        filename="images/BP/"+label+'-e-'+str(epoch)+"-m-"+model[0]
        print(label, "epoch:",epoch,"model:",model[0])
        accuracy,error=bpclass.execute_bp(model[1],X_train,X_test,y_train,y_test,label,epoch,filename,False)
        if(best_error==0 or error<best_error):
            best_error=error
            best_epoch=epoch
            best_model_name=model[0]
            best_model=model[1]
            best_accuracy=accuracy
print(label,"Best parameters:",best_model_name,best_epoch,best_accuracy, best_error)
filename="images/BP/"+label+'-e-'+str(best_epoch)+"-m-"+best_model_name
bpclass.execute_bp(best_model,X_train,X_test,y_train,y_test,label,best_epoch,filename,True)
warnings.resetwarnings()