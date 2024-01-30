from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc,mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
import sys
from plots import Plots_Class


class SVM_Class:

    def __init__(self):
        print('SVM')
        self.plotsClass = Plots_Class()

    def printPlots(self,df_plot,label,filename=""):
        self.plotsClass.printPlots(df_plot,label,filename="")
        
    def svm_classification(self, X_train,X_test,y_train,y_test,classification_kernel,constant,degree,label, plot=False):

        filename="images/SVM/"+label+"-"+classification_kernel+'-'+str(constant)+'-'+str(degree)
       
        # SVM classifiers
        svm_classifier = SVC(kernel=classification_kernel, C=constant, gamma='scale', random_state=42,degree=degree)
        
        #Train de model
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm_classifier.predict(X_test)    

        print(y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification error
        E = 100 * (conf_matrix[0][1] + conf_matrix[1][0]) / (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0])

        # Accuracy score (discrete values)
        accuracy = accuracy_score(y_test, y_pred)

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        
        if plot:

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            #Classification report
            classif_report=classification_report(y_test, y_pred)
            # Cross-validation 5-fold
            cross_val_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)
            # Calculate and print the mean cross-validation score
            mean_cv_score = cross_val_scores.mean()

            with open(filename+"-output.txt", 'w') as file:      
                sys.stdout = file  # Redirect stdout
                print("Cross-Validation Scores:", cross_val_scores)
                print("Mean Cross-Validation Score:", mean_cv_score)
                print(f"\nConfusion Matrix for {label} Dataset:")
                print(conf_matrix)
                print("Classification error: ", E)
                print(f"\nClassification Report for {label} Dataset:")
                print(classif_report)
                print("Mean Squared Separable Error:", mse)
                print("R-squared Separable:", r2)
                print("Accuracy:", accuracy*100)
                print(f'ROC AUC: {roc_auc}')
        
            sys.stdout = sys.__stdout__
            self.plotsClass.printPredictionPlots(X_test,y_test,y_pred,label,filename)
            self.plotsClass.plot_conf_matrix(conf_matrix,filename)
            self.plotsClass.plot_roc(fpr, tpr,roc_auc,filename)
        
        print("Classification error: ", E,"Accuracy:", accuracy*100)

        return accuracy*100, E
