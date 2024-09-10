import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, matthews_corrcoef
from feature_engine.creation import MathematicalCombinations
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures
import xgboost as xgb
import lightgbm as lgb
import shap
import lime
import lime.lime_tabular
import optuna
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
data = pd.read_csv('obesity_data.csv')

# Data Exploration
logging.info("Dataset Info:")
logging.info(data.info())

logging.info("\nFirst few rows of the dataset:")
logging.info(data.head())

logging.info("\nSummary Statistics:")
logging.info(data.describe(include='all'))

logging.info("\nMissing Values:")
logging.info(data.isnull().sum())

# Define features and target
X = data.drop('obesity_status', axis=1)  # Features
y = data['obesity_status']  # Target

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

# Custom Transformer for Advanced Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        self.transformer = PowerTransformer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X_poly = self.poly.fit_transform(X[numerical_features])
        X_poly_df = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(numerical_features))
        X_transformed = pd.concat([X, X_poly_df], axis=1)
        X_transformed[numerical_features] = self.transformer.fit_transform(X_transformed[numerical_features])
        return X_transformed

# Advanced Outlier Removal
class CustomOutlierRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (self.threshold * IQR)
            upper_bound = Q3 + (self.threshold * IQR)
            X = X[(X[column] >= lower_bound) & (X[column] <= upper_bound)]
        return X

# Feature Engineering
interaction = MathematicalCombinations(variables=numerical_features.tolist(), method='interaction')
X = interaction.fit_transform(X)

# Winsorize outliers
winsorizer = Winsorizer(capping_method='quantiles', tail='both', fold=1.5)
X[numerical_features] = winsorizer.fit_transform(X[numerical_features])

# Data Augmentation - Synthetic Data Generation
from sklearn.utils import resample

X_augment, y_augment = resample(X, y, n_samples=2*len(y), random_state=42)

# Preprocessing pipelines
categorical_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Full pipeline with additional models
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['xgb', 'lgb', 'rf', 'et', 'logreg'])
    
    if model_type == 'xgb':
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 50, 300),
            max_depth=trial.suggest_int('xgb_max_depth', 3, 15),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            random_state=42
        )
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int('lgb_n_estimators', 50, 300),
            max_depth=trial.suggest_int('lgb_max_depth', 3, 15),
            learning_rate=trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
            num_leaves=trial.suggest_int('lgb_num_leaves', 20, 100),
            subsample=trial.suggest_float('lgb_subsample', 0.6, 1.0),
            random_state=42
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 300),
            max_depth=trial.suggest_int('rf_max_depth', 3, 15),
            min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 10),
            random_state=42
        )
    elif model_type == 'et':
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int('et_n_estimators', 50, 300),
            max_depth=trial.suggest_int('et_max_depth', 3, 15),
            min_samples_split=trial.suggest_int('et_min_samples_split', 2, 10),
            random_state=42
        )
    else:
        model = LogisticRegression(
            penalty=trial.suggest_categorical('logreg_penalty', ['l1', 'l2']),
            C=trial.suggest_float('logreg_C', 1e-4, 1e4, log=True),
            solver=trial.suggest_categorical('logreg_solver', ['liblinear', 'saga']),
            random_state=42
        )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_eng', FeatureEngineering()),
        ('classifier', model)
    ])
    
    cv = StratifiedKFold(n_splits=5)
    score = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return score.mean()

# Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

logging.info("\nBest Trial:")
logging.info(study.best_trial)

# Best model
best_params = study.best_trial.params
model_type = best_params.pop('model_type')

if model_type == 'xgb':
    best_model = xgb.XGBClassifier(**best_params)
elif model_type == 'lgb':
    best_model = lgb.LGBMClassifier(**best_params)
elif model_type == 'rf':
    best_model = RandomForestClassifier(**best_params)
elif model_type == 'et':
    best_model = ExtraTreesClassifier(**best_params)
else:
    best_model = LogisticRegression(**best_params)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_eng', FeatureEngineering()),
    ('classifier', best_model)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the best model
pipeline.fit(X_train, y_train)

# Evaluate the best model
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

logging.info("\nModel Evaluation:")
logging.info("Accuracy: %f", accuracy_score(y_test, y_pred))
logging.info("ROC-AUC Score: %f", roc_auc_score(y_test, y_proba))
logging.info("F1 Score: %f", f1_score(y_test, y_pred))
logging.info("Matthews Correlation Coefficient (MCC): %f", matthews_corrcoef(y_test, y_pred))
logging.info("Classification Report:\n %s", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Obese', 'Obese'], yticklabels=['Not Obese', 'Obese'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, marker='.', label='Best Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(10, 7))
plt.plot(recall, precision, marker='.', label='Best Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('precision_recall_curve.png')
plt.show()

# SHAP Values
explainer = shap.Explainer(pipeline.named_steps['classifier'])
shap_values = explainer(pipeline.named_steps['preprocessor'].transform(X_test))

# Plot SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, pipeline.named_steps['preprocessor'].transform(X_test), feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out())
plt.savefig('shap_summary_plot.png')
plt.show()

# LIME for local interpretability
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=pipeline.named_steps['preprocessor'].transform(X_train),
    feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out(),
    class_names=['Not Obese', 'Obese'],
    mode='classification'
)

# Explain a single prediction
i = 0  # Index of the instance to explain
exp = explainer_lime.explain_instance(pipeline.named_steps['preprocessor'].transform(X_test)[i], pipeline.named_steps['classifier'].predict_proba)
exp.show_in_notebook(show_table=True, show_all=False)

# Save the model
joblib.dump(pipeline, 'obesity_model.pkl')

logging.info("\nModel saved to 'obesity_model.pkl'")

# Basic User Interface
import tkinter as tk
from tkinter import filedialog, messagebox

class ModelUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Obesity Prediction Model')
        
        self.label = tk.Label(root, text='Upload a CSV file for prediction:')
        self.label.pack()
        
        self.upload_btn = tk.Button(root, text='Upload File', command=self.upload_file)
        self.upload_btn.pack()
        
        self.predict_btn = tk.Button(root, text='Predict', command=self.predict, state=tk.DISABLED)
        self.predict_btn.pack()
        
        self.result_label = tk.Label(root, text='')
        self.result_label.pack()
        
    def upload_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
        if self.filepath:
            self.predict_btn.config(state=tk.NORMAL)
            messagebox.showinfo('File Upload', 'File successfully uploaded!')
    
    def predict(self):
        try:
            new_data = pd.read_csv(self.filepath)
            predictions = pipeline.predict(new_data)
            self.result_label.config(text=f'Predictions: {predictions}')
        except Exception as e:
            messagebox.showerror('Error', str(e))

# Run the UI
root = tk.Tk()
app = ModelUI(root)
root.mainloop()
