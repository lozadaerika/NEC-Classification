import matplotlib.pyplot as plt

class Plots_Class:

    def plot_pca(self,X_pca,y_train,title="",filename=""):
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', marker='o')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        plt.show()
        if filename!="":
            plt.savefig(filename+"-pca.png")
        else:
            plt.show()

    def plot_roc(self,fpr,tpr,roc_auc,filename=""):
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        if filename!="":
            plt.savefig(filename+"-roc.png")
        else:
            plt.show()

    def plot_conf_matrix(self,conf_matrix,filename=""):
        fig, ax = plt.subplots()
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=15)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0, 1], ['Class 0', 'Class 1'])
        plt.yticks([0, 1], ['Class 0', 'Class 1'])
        plt.colorbar(cax)
        if filename!="":
            plt.savefig(filename+"-conf-matrix.png")
        else:
            plt.show()

    def printPlots(self,df_plot,label,filename=""):
            # Visualize data
            plt.scatter(
                df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 0],
                df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 1],
                color='blue',
                label='Class 0',
                s=3  
            )
            plt.scatter(
                df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 0],
                df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 1],
                color='red',
                label='Class 1',
                s=3  
            )
            plt.title(f'Scatter Plot of {label} Dataset')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            if filename!="":
                plt.savefig(filename+"-plot.png")
            else:
                plt.show()
        
    def printPredictionPlots(self,X_test1,y_test1,y_pred1,label1,filename=""):
        # Visualize data
        plt.scatter(
            X_test1[y_test1 == 0][:, 0],
            X_test1[y_test1 == 0][:, 1],
            color='blue',
            label='True Class 0',
            s=3  
        )
        plt.scatter(
            X_test1[y_test1 == 1][:, 0],
            X_test1[y_test1 == 1][:, 1],
            color='red',
            label='True Class 1',
            s=3  
        )
        plt.scatter(
            X_test1[y_pred1 == 0][:, 0],
            X_test1[y_pred1 == 0][:, 1],
            facecolors='none',
            edgecolors='blue',
            label=f'Predicted Class 0 ({label1}])',
            s=30  
        )
        plt.scatter(
            X_test1[y_pred1 == 1][:, 0],
            X_test1[y_pred1 == 1][:, 1],
            facecolors='none',
            edgecolors='red',
            label=f'Predicted Class 1 ({label1})',
            s=30  
        )
        plt.title(f'Scatter Plot of Test Set Predictions ({label1}])')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        if filename!="":
            plt.savefig(filename+"-predictions.png")
        else:
            plt.show()