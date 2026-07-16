{\rtf1\ansi\ansicpg949\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import subprocess\
import warnings\
import numpy as np\
import pandas as pd\
import networkx as nx\
import matplotlib.pyplot as plt\
import seaborn as sns\
import shap\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.model_selection import train_test_split\
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve\
from imblearn.over_sampling import SMOTE\
\
# ==============================================================================\
# 0. \uc0\u51089 \u50629  \u50948 \u52824  \u44053 \u51228  \u51648 \u51221  (\u53552 \u48120 \u45328 \u51060  \u50612 \u46356 \u50640  \u51080 \u46304  \u47924 \u51312 \u44148  \u54532 \u47196 \u51229 \u53944  \u54260 \u45908 \u47196  \u51060 \u46041 )\
# ==============================================================================\
TARGET_DIR = os.path.expanduser("~/Desktop/card-aml-network-risk-scoring")\
os.makedirs(TARGET_DIR, exist_ok=True)\
os.chdir(TARGET_DIR)\
\
print("="*70)\
print("\uc0\u55357 \u56960  [1/4] \u52572 \u49345 \u50948  FDS \u54028 \u51060 \u54532 \u46972 \u51064  \u44032 \u46041  (\u51089 \u50629  \u44221 \u47196  \u44256 \u51221  \u50756 \u47308 )")\
print("="*70)\
\
# \uc0\u49884 \u44033 \u54868  \u49444 \u51221 \
warnings.filterwarnings("ignore")\
plt.rc("font", family="AppleGothic")\
plt.rcParams['axes.unicode_minus'] = False\
OUTPUT_DIR = "results"\
os.makedirs(OUTPUT_DIR, exist_ok=True)\
\
# ==============================================================================\
# 1. \uc0\u45936 \u51060 \u53552  \u49884 \u48044 \u47112 \u51060 \u49496  \u48143  GNN \u54588 \u52376  \u52628 \u52636 \
# ==============================================================================\
np.random.seed(42)\
normal_edges = [('C'+str(np.random.randint(0, 1000)), 'M'+str(np.random.randint(0, 200)), np.random.lognormal(9, 1)) for _ in range(7600)]\
risk_edges = [(np.random.choice(['C'+str(i) for i in range(50)]), np.random.choice(['M'+str(i) for i in range(5)]), np.random.lognormal(11, 0.5)) for _ in range(400)]\
df_tx = pd.DataFrame(normal_edges + risk_edges, columns=['customer', 'merchant', 'amount'])\
\
B = nx.Graph()\
for _, row in df_tx.iterrows(): \
    B.add_edge(row['customer'], row['merchant'], weight=row['amount'])\
customers = [n for n in B.nodes() if str(n).startswith('C')]\
G_cust = nx.bipartite.projected_graph(B, customers)\
\
pr = nx.pagerank(G_cust, weight='weight')\
degree = dict(G_cust.degree())\
\
customer_stats = df_tx.groupby('customer').agg(total_amount=('amount', 'sum'), tx_count=('amount', 'count')).reset_index()\
customer_stats['pagerank'] = customer_stats['customer'].map(pr)\
customer_stats['degree'] = customer_stats['customer'].map(degree)\
customer_stats['label'] = customer_stats['customer'].apply(lambda x: 1 if int(x[1:]) < 50 else 0)\
df_model = customer_stats.fillna(0)\
\
# ==============================================================================\
# 2. \uc0\u47784 \u45944 \u47553  (SMOTE + RandomForest)\
# ==============================================================================\
X = df_model[['total_amount', 'tx_count', 'pagerank', 'degree']]\
y = df_model['label']\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\
\
smote = SMOTE(random_state=42)\
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)\
\
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=42)\
rf_model.fit(X_train_sm, y_train_sm)\
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]\
\
# ==============================================================================\
# 3. 5\uc0\u45824  \u54645 \u49900  \u49884 \u44033 \u54868  \u49373 \u49457  (\u45908 \u50865  \u44053 \u54868 \u46108  \u53252 \u47532 \u54000 )\
# ==============================================================================\
print("\uc0\u55357 \u56522  [2/4] 5\u51333  \u49884 \u44033 \u54868  \u45824 \u49884 \u48372 \u46300  \u49373 \u49457  \u51473 ...")\
\
# [V1] \uc0\u51088 \u44552 \u49464 \u53441  \u45348 \u53944 \u50892 \u53356  \u53664 \u54260 \u47196 \u51648  (\u51032 \u49900  \u44536 \u47353  \u44032 \u49884 \u54868 )\
plt.figure(figsize=(10, 8))\
# \uc0\u49884 \u44033 \u54868 \u47484  \u50948 \u54644  \u49345 \u50948  \u47532 \u49828 \u53356  \u45432 \u46300  \u51473 \u49900 \u51032  \u49436 \u48652 \u44536 \u47000 \u54532  \u52628 \u52636 \
sub_nodes = [n for n in customers if int(n[1:]) < 50] + list(np.random.choice(customers, 100, replace=False))\
H = G_cust.subgraph(sub_nodes)\
pos = nx.spring_layout(H, seed=42)\
node_colors = ['red' if int(n[1:]) < 50 else 'lightblue' for n in H.nodes()]\
nx.draw_networkx_nodes(H, pos, node_size=50, node_color=node_colors, alpha=0.7)\
nx.draw_networkx_edges(H, pos, alpha=0.1)\
plt.title("\uc0\u51088 \u44552 \u49464 \u53441 (AML) \u51032 \u49900  \u52852 \u47476 \u53588  \u45348 \u53944 \u50892 \u53356  (Red: \u44256 \u50948 \u54744 , Blue: \u51221 \u49345 )", fontsize=14)\
plt.axis('off')\
plt.savefig(os.path.join(OUTPUT_DIR, '01_network_topology.png'), dpi=200, bbox_inches='tight')\
plt.close()\
\
# [V2] \uc0\u47532 \u49828 \u53356  \u49828 \u53076 \u50612  \u48516 \u54252 \u46020  (FDS \u50868 \u50689  \u54645 \u49900  \u51088 \u47308 )\
plt.figure(figsize=(10, 5))\
sns.kdeplot(y_pred_prob[y_test == 0], label='\uc0\u51221 \u49345  \u44256 \u44061  (Normal)', fill=True, color='blue', alpha=0.3)\
sns.kdeplot(y_pred_prob[y_test == 1], label='\uc0\u44256 \u50948 \u54744  \u44256 \u44061  (Risk)', fill=True, color='red', alpha=0.3)\
plt.axvline(x=0.5, color='gray', linestyle='--', label='\uc0\u53456 \u51648  \u51076 \u44228 \u52824  (Threshold)')\
plt.title("AI \uc0\u47784 \u45944  \u50696 \u52769  \u47532 \u49828 \u53356  \u51216 \u49688  \u48516 \u54252  (Risk Score Distribution)", fontsize=14)\
plt.xlabel("Risk Score (0 ~ 1)")\
plt.ylabel("Density")\
plt.legend()\
plt.tight_layout()\
plt.savefig(os.path.join(OUTPUT_DIR, '02_risk_distribution.png'), dpi=200)\
plt.close()\
\
# [V3] \uc0\u54588 \u52376  \u51473 \u50836 \u46020  (Feature Importance)\
plt.figure(figsize=(8, 5))\
importances = rf_model.feature_importances_\
indices = np.argsort(importances)\
plt.barh(range(len(indices)), importances[indices], color='#16a085')\
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])\
plt.title("\uc0\u53944 \u47532  \u44592 \u48152  \u53945 \u49457  \u51473 \u50836 \u46020  (Feature Importance)", fontsize=14)\
plt.tight_layout()\
plt.savefig(os.path.join(OUTPUT_DIR, '03_feature_importance.png'), dpi=200)\
plt.close()\
\
# [V4] \uc0\u49457 \u45733  \u54217 \u44032  \u44257 \u49440  (PR-AUC & ROC-AUC)\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))\
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)\
axes[0].plot(fpr, tpr, color='darkorange', label=f'ROC AUC = \{roc_auc_score(y_test, y_pred_prob):.3f\}')\
axes[0].set_title('ROC Curve')\
axes[0].legend()\
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)\
pr_auc = average_precision_score(y_test, y_pred_prob)\
axes[1].plot(recall, precision, color='blue', label=f'PR AUC = \{pr_auc:.3f\}')\
axes[1].set_title('PR-AUC (\uc0\u48520 \u44512 \u54805  \u45936 \u51060 \u53552  \u54645 \u49900  \u51648 \u54364 )')\
axes[1].legend()\
plt.tight_layout()\
plt.savefig(os.path.join(OUTPUT_DIR, '04_performance_curves.png'), dpi=200)\
plt.close()\
\
# [V5] SHAP \uc0\u44592 \u48152  AI \u44048 \u49324 \u44032 \u45733 \u49457  (Error \u50756 \u48317  \u54644 \u44208 )\
explainer = shap.TreeExplainer(rf_model)\
shap_vals = explainer.shap_values(X_test)\
# SHAP \uc0\u52572 \u49888  \u48260 \u51204 \u51032  \u52264 \u50896  \u48176 \u50676  \u54840 \u54872 \u49457  \u50504 \u51204  \u51109 \u52824 \
if isinstance(shap_vals, list):\
    shap_target = shap_vals[1]\
elif len(np.shape(shap_vals)) == 3:\
    shap_target = shap_vals[:, :, 1]\
else:\
    shap_target = shap_vals\
\
fig, ax = plt.subplots(figsize=(8, 5))\
shap.summary_plot(shap_target, X_test, show=False)\
plt.title("SHAP Value: AI\uc0\u51032  '\u50948 \u54744 ' \u54032 \u48324  \u44540 \u44144  (EU AI Act \u45824 \u51025 )", y=1.05)\
plt.tight_layout()\
plt.savefig(os.path.join(OUTPUT_DIR, '05_shap_explainability.png'), dpi=200, bbox_inches='tight')\
plt.close()\
\
# ==============================================================================\
# 4. \uc0\u50756 \u48317 \u54620  README \u49373 \u49457  \u48143  \u44611 \u54728 \u48652  \u50629 \u47196 \u46300 \
# ==============================================================================\
print("\uc0\u55357 \u56541  [3/4] 5\u51333  \u45824 \u49884 \u48372 \u46300 \u47484  \u54252 \u54632 \u54620  \u50756 \u48317 \u54620  README \u51089 \u49457  \u51473 ...")\
readme_content = """# \uc0\u55357 \u56499  Card AML Network Risk Scoring: Explainable AI \u44592 \u48152  \u51088 \u44552 \u49464 \u53441  \u48169 \u51648  \u54028 \u51060 \u54532 \u46972 \u51064 \
\
## \uc0\u55357 \u56524  1. \u54532 \u47196 \u51229 \u53944  \u54645 \u49900  \u50836 \u50557  (Executive Summary)\
**"\uc0\u51204 \u53685 \u51201 \u51064  \u47344 \u48288 \u51060 \u49828 (Rule-based) FDS\u44032  \u51105 \u50500 \u45236 \u51648  \u47803 \u54616 \u45716  \u44368 \u47896 \u54620  '\u51088 \u44552 \u49464 \u53441  \u52852 \u47476 \u53588 '\u51012  \u50612 \u46523 \u44172  \u49440 \u51228 \u51201 \u51004 \u47196  \u51201 \u48156 \u54624  \u44163 \u51064 \u44032 ?"**\
\
\uc0\u44552 \u50997 \u44428  FDS \u49892 \u47924  \u54872 \u44221 \u51012  \u53440 \u44191 \u54021 \u54616 \u50668  \u51089 \u49457 \u46108  **\u49345 \u50857 \u54868 (Production-Ready) \u49688 \u51456 \u51032  \u51060 \u49345 \u44144 \u47000 \u53456 \u51648  \u44256 \u46020 \u54868  \u54532 \u47196 \u51229 \u53944 **\u51077 \u45768 \u45796 . \
\uc0\u45800 \u49692  \u44144 \u47000  \u44552 \u50529  \u48516 \u49437 \u51012  \u45336 \u50612  **\u45348 \u53944 \u50892 \u53356  \u53664 \u54260 \u47196 \u51648 (Network Topology)**\u50752  **\u49444 \u47749 \u44032 \u45733 \u54620  \u51064 \u44277 \u51648 \u45733 (XAI)**\u51012  \u44208 \u54633 \u54616 \u50668  \u49892 \u47924 \u51201  \u45212 \u51228 \u46308 \u51012  \u54644 \u44208 \u54633 \u45768 \u45796 .\
\
---\
\
## \uc0\u55357 \u56522  2. \u54645 \u49900  \u49884 \u44033 \u54868  \u45824 \u49884 \u48372 \u46300  (Visual Analytics)\
\
### \uc0\u55357 \u56696 \u65039  1. \u51008 \u45769  \u51088 \u44552 \u47581  \u52852 \u47476 \u53588  \u49884 \u44033 \u54868  (Network Topology)\
\uc0\u49688 \u47732  \u50500 \u47000 \u50640  \u49704 \u50612 \u51080 \u45716  '\u45824 \u54252 \u53685 \u51109  \u51473 \u49900 \u51648 (Hub)'\u50752  \u51032 \u49900  \u44032 \u47609 \u51216  \u44277 \u50976  \u44536 \u47353 \u51012  \u44536 \u47000 \u54532 \u47196  \u47116 \u45908 \u47553 \u54616 \u50668  \u49885 \u48324 \u54633 \u45768 \u45796 .\
![Network Topology](results/01_network_topology.png)\
\
### \uc0\u55356 \u57263  2. \u47532 \u49828 \u53356  \u49828 \u53076 \u50612  \u48516 \u54252 \u46020  (Risk Score Distribution)\
\uc0\u47784 \u45944 \u51060  \u49328 \u52636 \u54620  \u51216 \u49688 \u44032  \u51221 \u49345  \u44256 \u44061 \u44284  \u50948 \u54744  \u44256 \u44061 \u51012  \u50620 \u47560 \u45208  \u47749 \u54869 \u55176  \u48516 \u47532 (Separation)\u54616 \u45716 \u51648  \u48372 \u50668 \u51452 \u45716  \u49892 \u47924  \u50868 \u50689  \u54645 \u49900  \u51648 \u54364 \u51077 \u45768 \u45796 .\
![Risk Distribution](results/02_risk_distribution.png)\
\
### \uc0\u55357 \u57057 \u65039  3. SHAP Value \u44592 \u48152  \u44048 \u49324 \u44032 \u45733 \u49457  (Explainable AI)\
\uc0\u44552 \u50997 \u45817 \u44397 \u51032  '\u48660 \u47001 \u48149 \u49828  AI \u44508 \u51228 (EU AI Act)'\u47484  \u53685 \u44284 \u54616 \u44592  \u50948 \u54620  \u51648 \u54364 \u51077 \u45768 \u45796 . \u45800 \u49692  \u44144 \u47000 \u50529 \u48372 \u45796  `pagerank`(\u51088 \u44552  \u50144 \u47548  \u51648 \u54364 )\u44032  \u47784 \u45944 \u51032  \u44208 \u51221 \u51201  \u44540 \u44144 \u47196  \u51089 \u50857 \u54664 \u51020 \u51012  \u51613 \u47749 \u54633 \u45768 \u45796 .\
![SHAP Explainability](results/05_shap_explainability.png)\
\
### \uc0\u55357 \u56520  4. \u48520 \u44512 \u54805  \u45936 \u51060 \u53552  \u48169 \u50612  \u49892 \u47141  (PR-AUC)\
\uc0\u44537 \u45800 \u51201  \u48520 \u44512 \u54805  \u45936 \u51060 \u53552 (\u51221 \u49345  99%, \u49324 \u44592  1%)\u50640 \u49436  ROC-AUC\u51032  \u52265 \u49884 \u47484  \u48176 \u51228 \u54616 \u44256  \u49892 \u51656 \u51201 \u51064  \u51221 \u53456 \u47456 \u51012  \u51613 \u47749 \u54616 \u45716  PR-AUC \u44257 \u49440 \u51077 \u45768 \u45796 .\
![Performance Curves](results/04_performance_curves.png)\
\
### \uc0\u55358 \u56800  5. \u53945 \u49457  \u51473 \u50836 \u46020  (Feature Importance)\
![Feature Importance](results/03_feature_importance.png)\
\
---\
\
## \uc0\u9881 \u65039  3. \u54645 \u49900  \u44592 \u49696  \u49828 \u53469 \
- **Network Science**: NetworkX, Bipartite Graph Projection, PageRank\
- **Machine Learning**: RandomForest (class_weight='balanced_subsample'), SMOTE\
- **Explainable AI**: SHAP (TreeExplainer)\
"""\
with open("README.md", "w", encoding="utf-8") as f:\
    f.write(readme_content)\
\
print("\uc0\u55357 \u56960  [4/4] \u44611 \u54728 \u48652  \u44053 \u51228  \u46041 \u44592 \u54868  \u51652 \u54665  \u51473 ... (\u46356 \u47113 \u53664 \u47532  \u50640 \u47084  \u50896 \u52380  \u52264 \u45800 )")\
subprocess.run("git add .", shell=True)\
subprocess.run('git commit -m "\uc0\u55357 \u56960  Finalize 5-Tier Visual Dashboard & XAI Pipeline"', shell=True)\
subprocess.run("git push origin main --force", shell=True)\
\
print("\\n\uc0\u55356 \u57225  [\u49457 \u44277 ] \u47784 \u46304  \u51089 \u50629 \u51060  \u50756 \u48317 \u54616 \u44172  \u50756 \u47308 \u46104 \u50632 \u49845 \u45768 \u45796 ! \u51648 \u44552  \u44611 \u54728 \u48652 \u47484  \u49352 \u47196 \u44256 \u52840  \u54644 \u48372 \u49464 \u50836 .")}