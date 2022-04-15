import itertools
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def model_history(model_history):

    #ploting 2 plots on horizontal axis
    fig,(ax1,ax2) =  plt.subplots(1,2,figsize=(16,8))
    
    # summarize history for accuracy
    ax1.plot(model_history.history['accuracy'],c ="darkblue")
    ax1.plot(model_history.history['val_accuracy'],c ="crimson")
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper right')

    # summarize history for loss
    ax2.plot(model_history.history['loss'],c ="darkblue")
    ax2.plot(model_history.history['val_loss'],c ="crimson")
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper right')
    
    fig.suptitle("Model History")

#Prints a classifcation report with accuracy below it
def c_report(y_true,y_pred,target_names=[]):
    print("Classifictaion Report")
    print(classification_report(y_true, y_pred, target_names=target_names))
    acc_scr = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_scr))

#plots confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    #give blueish color mapping
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    