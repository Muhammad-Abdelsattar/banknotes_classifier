import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import json

def write_metrics(metrics_dict,file):
    dir = os.path.dirname(file)
    os.makedirs(dir,exist_ok=True)
    with open(file,"w") as f:
        json.dump(metrics_dict,f)

def write_confusion_matrix_plot(confusion_matrix,file):
    display_labels = ["1f", "1b", "5f", "5b", "10f", "10b", "20f", "20b", "50f", "50b", "100f", "100b", "200f", "200b"]
    dir = os.path.dirname(file)
    os.makedirs(dir,exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.6, 15.6))
    ConfusionMatrixDisplay(confusion_matrix.astype(np.int8),display_labels=display_labels).plot(
        ax=ax, colorbar=False, include_values=True
    )
    ax.set_title('Confusion Matrix')
    fig.savefig(file,bbox_inches='tight')


