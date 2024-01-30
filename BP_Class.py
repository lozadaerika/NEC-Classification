import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc,mean_squared_error, r2_score
from plots import Plots_Class



import sys


class BP_Class:

    def __init__(self):
        self.plotsClass = Plots_Class()
        

    def printPlots(self,df_plot,label):
        self.plotsClass.printPlots(df_plot,label)

    def execute_bp(self,model,X_train,X_test,y_train,y_test,label,epoch,filename,plot):
    
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)

        threshold=0.5
       
        # Train the model
        verbose=0
        if(plot):
            verbose=2
        model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_test, y_test),verbose=verbose)

        # Predict the test data and binarize
        y_pred = model.predict(X_test,verbose=verbose)
        y_pred = (y_pred > threshold).astype(int)
        y_test = (y_test > threshold).astype(int)

        y_pred_aux=y_test.copy()
        flat_list = [item for sublist in y_pred for item in sublist]
        for i in range(len(y_test)):
            y_pred_aux[i]=flat_list[i]
        y_pred = y_pred_aux

        # Compile the model using cross-validation 5-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        loss_per_fold = []
        fold_no = 1
        for train_index, val_index in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            # Train the model
            history = model.fit(X_train_fold, y_train_fold, epochs=epoch, batch_size=32, validation_data=(X_val_fold, y_val_fold),verbose=0)
            scores = model.evaluate(X_train_fold, y_train_fold, verbose=0)
            loss_per_fold.append(scores[0])
        # Increase fold number
            fold_no = fold_no + 1

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Separable Error:", mse)

        # Accuracy score (discrete values)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)


        if plot:
            r2 = r2_score(y_test, y_pred)
            print("R-squared Separable:", r2)
                      
            # Evaluate the model on the test set
            loss, accuracy = model.evaluate(X_test, y_test)
            print('Test loss:', loss)
            print('Test accuracy:', accuracy)

            print("")
            print("Cross-Validation:")
            print('Loss per fold:')
            for i in range(0, len(loss_per_fold)):
                print(f'{loss_per_fold[i]}', end=' ')
            print("")    
            print('Average Loss :',np.mean(loss_per_fold))
            print("")  

            self.plotsClass.printPredictionPlots(X_test,y_test,y_pred,label,filename)

            # Plot the confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix for Dataset:")
            print(conf_matrix)
            self.plotsClass.plot_conf_matrix(conf_matrix,filename)

            # Classification error
            E = 100 * (conf_matrix[0][1] + conf_matrix[1][0]) / (conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[0][1] + conf_matrix[1][0])
            print("Classification error: ", E)

            # Classification report
            print(f"\nClassification Report for Dataset:")
            classif_report=classification_report(y_test, y_pred)
            print(classif_report)

            # ROC curve and AUC (discrete values)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            print(f'ROC AUC: {roc_auc}')
            self.plotsClass.plot_roc(fpr,tpr,roc_auc,filename)

            # Plot PCA of predicted data
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_test)
            self.plotsClass.plot_pca(X_pca,y_pred,'PCA of predicted data',filename)

            with open(filename+"-output.txt", 'w') as file:      
                sys.stdout = file  # Redirect stdout
                print("Cross-Validation Scores:", loss_per_fold)
                print("Mean Cross-Validation Score:", np.mean(loss_per_fold))
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

        return accuracy,mse