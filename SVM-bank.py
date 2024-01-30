from SVM_Method import SVM_Class  
import pandas as pd
from sklearn.model_selection import train_test_split

#Kernel list
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
svm_constant_min=1
svm_constant_max=10
svm_constant_step=1
svm_degree_min=1
svm_degree_max=4
svm_degree_step=1

min_error_aux=1

# Load the datasets
file_path = 'A2-bank/bank-additional-processed.csv'

# Load data into DataFrames
df_dataset = pd.read_csv(file_path, delimiter=',', header=1)

# Display the first few rows of each dataset
print(df_dataset.head())

# Convert the last column to integers
df_dataset[df_dataset.columns[-1]] = df_dataset[df_dataset.columns[-1]].astype(int)

#print(df_dataset)

# Extract features and labels
X_train, X_test, y_train, y_test  = train_test_split(
    df_dataset.iloc[:, :-1].values,
    df_dataset.iloc[:, -1].values,
    test_size=0.2,
    random_state=42
)

svmclass = SVM_Class()

svmclass.printPlots(df_dataset,'Bank')

best_error=0
best_kernel=""
best_constant=0
best_degree=0
best_accuracy=0

try:
    for kernel in svm_kernels:
        for constant in range(svm_constant_min, svm_constant_max +1 , svm_constant_step):
            for degree in range(svm_degree_min, svm_degree_max +1 , svm_degree_step):
                print(f"Bank PARAM> kernel={kernel} C={constant}","Degree",degree)
                accuracy, E=svmclass.svm_classification(X_train,X_test,y_train, y_test,kernel,constant,degree,'Bank',False)
                if(best_error==0 or E<best_error):
                    best_error=E
                    best_kernel=kernel
                    best_constant=constant
                    best_degree=degree
                    best_accuracy=accuracy
    print("Bank Best parameters:",best_kernel,best_constant,best_degree, best_accuracy, best_error)
    svmclass.svm_classification(X_train,X_test,y_train, y_test,best_kernel,best_constant,best_degree,'Bank',True)
except ValueError as e:
    # Code to handle the exception
    print("Bank Best parameters:",best_kernel,best_constant,best_degree, best_accuracy, best_error)
    svmclass.svm_classification(X_train,X_test,y_train, y_test,best_kernel,best_constant,best_degree,'Bank',True)
    print(f"Error: {e}")
except Exception as e:
    # Code to handle other exceptions
    print("Bank Best parameters:",best_kernel,best_constant,best_degree, best_accuracy, best_error)
    svmclass.svm_classification(X_train,X_test,y_train, y_test,best_kernel,best_constant,best_degree,'Bank',True)
    print(f"Unexpected error: {e}")
