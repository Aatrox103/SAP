import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

plt.rcParams['font.size'] = 15

with open('data/gpt_negative_score.json', mode='r', encoding="utf8") as f:
    negative_samples = json.load(f)

with open('data/gpt_positive_score.json', mode='r', encoding="utf8") as f:
    positive_samples = json.load(f)

labels1 = np.concatenate([np.ones(50), np.zeros(50)])
scores1 = np.concatenate([negative_samples, positive_samples])

with open('data/p_api_negative_score.json', mode='r', encoding="utf8") as f:
    negative_samples = json.load(f)

with open('data/p_api_positive_score.json', mode='r', encoding="utf8") as f:
    positive_samples = json.load(f)

labels2 = np.concatenate([np.ones(50), np.zeros(50)])
scores2 = np.concatenate([negative_samples, positive_samples])


fpr1, tpr1, thresholds1 = roc_curve(labels1, scores1)
fpr2, tpr2, thresholds2 = roc_curve(labels2, scores2)


roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)


plt.figure()
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='GPT3.5 (AUC = %0.2f)' % roc_auc1,linestyle='solid')
plt.plot(fpr2, tpr2, color='green', lw=2, label='Perspective api (AUC = %0.2f)' % roc_auc2,linestyle='dashed')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()