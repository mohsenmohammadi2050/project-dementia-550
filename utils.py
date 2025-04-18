import re
import os
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("averaged_perceptron_tagger")
from nltk.util import ngrams
import spacy
import string
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_validate

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Extract and clean meta data
def clean_meta_data(dataset, test_data=True):
    dataset = dataset.copy()
    if "mmse" in dataset.columns:
        dataset.drop(columns=["mmse"], inplace=True)
    dataset.columns = [column.strip() for column in dataset.columns]
    for column in dataset.columns:
        dataset[column] = dataset[column].apply(lambda x:
            np.nan if pd.isna(x) else x.strip()
            if isinstance(x, str) else x)
    if test_data:
        dataset["gender"] = dataset["gender"].apply(lambda x: 1 if x=="male" else 0)
    dataset.index = dataset["ID"]
    dataset.drop(columns=["ID"], inplace=True)
    return dataset

# Extract speech fron transcripts
def extract_speech_from_cha(file_path):
    speech_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("*PAR:"):
                clean_line = re.sub(r"\x15.*?\x15", "", line[5:]).strip()
                speech_data.append(clean_line)
    return speech_data

# Extract all sentences
def extract_all_sentences(path):
    all_sentences = []
    for filename in os.listdir(path):
        if filename.endswith(".cha"):
            file_path = os.path.join(path, filename)
            speech_data = extract_speech_from_cha(file_path)
            all_sentences.append(speech_data)
    return all_sentences


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Corrects spelling using TextBlob
def correct_spelling(sentence):
    return str(TextBlob(sentence).correct())

# Lemmatizes the word based on its part of speech using WordNetLemmatizer
def lemmatize_word(word):
    pos_tagged = pos_tag([word])[0][1]
    if pos_tagged.startswith('NN'):  # Nouns
        return lemmatizer.lemmatize(word, pos='n')
    elif pos_tagged.startswith('VB'):  # Verbs
        return lemmatizer.lemmatize(word, pos='v')
    elif pos_tagged.startswith('JJ'):  # Adjectives
        return lemmatizer.lemmatize(word, pos='a')
    else:
        return word

def clean_text(sentence):
    # Tokenizes and cleans the sentence by removing stopwords, non-alphabetic words,
    # punctuation, and applies lemmatization and spelling correction.
    sentence = correct_spelling(sentence.lower())  # Correct spelling first
    tokens = word_tokenize(sentence)
    # Remove non-alphabetic words, stopwords, and punctuation
    filtered_tokens = [lemmatize_word(word) for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_tokens)

# Initialize all ML models
def all_models(lg_params=None, svc_params=None, xgb_params=None, rf_params=None, random_state=42):  
    if lg_params:  
        log_reg = LogisticRegression(**lg_params, max_iter=5000, random_state=random_state)
    else:
        log_reg = LogisticRegression(max_iter=5000, random_state=random_state)
        
    if svc_params:
        svc = SVC(**svc_params, probability=True, random_state=random_state)
    else:
        svc = SVC(probability=True, random_state=random_state)
        
    if xgb_params:
        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **xgb_params,
            random_state=random_state)
        
    else:
        xgb_clf = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state)
            
        
    if rf_params:
        rf_clf = RandomForestClassifier(**rf_params, random_state=random_state)
    else:
        rf_clf = RandomForestClassifier(random_state=random_state)

    # Combine using VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[
            ('logreg', log_reg),
            ('rf', rf_clf),
            ('xgb', xgb_clf),
            ('svc', svc),
        ],
        voting='soft',
    )

    # Add to existing classifiers dictionary
    classifiers = {
        "XGBoost": xgb_clf,
        "Logistic Regression": log_reg,
        "SVM": svc,  # probability=True needed for soft voting
        "Random Forest": rf_clf,
        "Voting": voting_clf
    }
    
    return classifiers


# Get metrics for models
def get_model_metrics(models, X_train, y_train, X_test, y_test):
    metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "AUC": []
    }

    for name, model in models.items():
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)

        # Use predict_proba or decision_function for AUC
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = None

        metrics["Model"].append(name)
        metrics["Accuracy"].append(round(accuracy_score(y_test, y_pred), 3))
        metrics["Precision"].append(round(precision_score(y_test, y_pred), 3))
        metrics["Recall"].append(round(recall_score(y_test, y_pred), 3))
        metrics["F1 Score"].append(round(f1_score(y_test, y_pred), 3))

        if y_scores is not None:
            metrics["AUC"].append(round(roc_auc_score(y_test, y_scores), 3))
        else:
            metrics["AUC"].append("N/A")  # In case AUC can't be computed

    return metrics


# Get cross-validation metrics for models
def get_crossvalidation_metrics(models, X, y, cv=5):
    cv_results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "AUC": []
    }

    for name, clf in models.items():  
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        try:
            scores = cross_validate(
                clf, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False
            )

            cv_results["Model"].append(name)
            cv_results["Accuracy"].append(round(np.mean(scores['test_accuracy']), 3))
            cv_results["Precision"].append(round(np.mean(scores['test_precision']), 3))
            cv_results["Recall"].append(round(np.mean(scores['test_recall']), 3))
            cv_results["F1 Score"].append(round(np.mean(scores['test_f1']), 3))
            cv_results["AUC"].append(round(np.mean(scores['test_roc_auc']), 3))
        except Exception:
            # In case the model doesn't support probability estimates or scoring
            cv_results["Model"].append(name)
            cv_results["Accuracy"].append("N/A")
            cv_results["Precision"].append("N/A")
            cv_results["Recall"].append("N/A")
            cv_results["F1 Score"].append("N/A")
            cv_results["AUC"].append("N/A")
    
    return cv_results



# Plot metrics table
def plot_metrics_table(metrics_dict, title="Evaluation Metrics"):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(metrics_dict.keys()), fill_color='paleturquoise', align='left'),
        cells=dict(values=[metrics_dict[key] for key in metrics_dict], fill_color='lavender', align='left'))
    ])
    fig.update_layout(title=title)
    fig.show()


# Plot confusion matrices for models
def plot_confusion_matrices(models, X_train, y_train, X_test, y_test, title="Confusion Matrices (1: dementia, 0:non-dementia)"):
    subplot_titles = list(models.keys())
    fig = make_subplots(rows=1, cols=5, subplot_titles=subplot_titles)

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        heatmap = go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale='Blues',
            showscale=False,
            text=cm,  # Add the confusion matrix values as text
            texttemplate="%{text}",  # Format the text inside each cell
            colorbar=dict(title="Count")
        )

        fig.add_trace(heatmap, row=1, col=idx + 1)

    fig.update_layout(title_text=title, height=400, width=1500)
    fig.show()


# Plot confusion matrices with ROC curves
def plot_confusion_matrices_with_roc(models, X_train, y_train, X_test, y_test,
                                      title="Confusion Matrices & ROC Curves (1: dementia, 0: non-dementia)"):
    num_models = len(models)
    subplot_titles = list(models.keys())
    fig = make_subplots(
        rows=2, cols=num_models,
        subplot_titles=subplot_titles * 2,
        vertical_spacing=0.2
    )

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        heatmap = go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}"
        )
        fig.add_trace(heatmap, row=1, col=idx + 1)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)

        roc_curve_plot = go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} AUC = {auc_score:.2f}',
            line=dict(color='blue')
        )

        random_line = go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        )

        fig.add_trace(roc_curve_plot, row=2, col=idx + 1)
        fig.add_trace(random_line, row=2, col=idx + 1)

        # Label axes for ROC only (bottom row)
        fig.update_xaxes(title_text="False Positive Rate", row=2, col=idx + 1)
        fig.update_yaxes(title_text="True Positive Rate", row=2, col=idx + 1)

        # --- Add AUC text annotation in bottom-right of ROC plot ---
        fig.add_annotation(
            x=0.95, y=0.05,
            text=f"AUC = {auc_score:.2f}",
            showarrow=False,
            xref=f"x{idx+1+num_models if idx != 0 else ''}",
            yref=f"y{idx+1+num_models if idx != 0 else ''}",
            font=dict(size=12, color="black"),
            align="right",
            row=2, col=idx + 1
        )

    fig.update_layout(
        title_text=title,
        height=800,
        width=300 * num_models,
        showlegend=False,
        template='plotly_white'
    )
    fig.show()



# Plot bar plots for comparison
def plot_metrics_comparison(metrics_before, metrics_after, title=""):
    metrics_before = metrics_before.copy()
    metrics_after = metrics_after.copy()
    
    models = metrics_before.pop('Model')
    metrics_after.pop('Model')
    before_tuning = metrics_before
    after_tuning = metrics_after
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    
    # Create a 1-row, 5-column subplot
    fig = make_subplots(rows=1, cols=5, subplot_titles=titles)

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(name="Before Tuning", x=models, y=before_tuning[metric]),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Bar(name="After Tuning", x=models, y=after_tuning[metric]),
            row=1, col=i+1
        )

    fig.update_layout(
        title=f"Model Performance Comparison Before and After Tuning {title}",
        barmode='group',
        height=500,
        width=1600,
        legend=dict(x=1.05, y=1.1)
    )

    fig.show()


# TSNE visualization
def plot_tsne(X_2d, y, title="t-SNE Visualization"):
    label_map = {0: 'CC', 1: 'CD'}
    labels_mapped = [label_map[val] for val in y]

    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'label': labels_mapped
    })

    fig = px.scatter(
        df, x='x', y='y', color='label',
        title=title,
        labels={'label': 'Class'},
        color_discrete_map={'CC': 'blue', 'CD': 'red'}
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title="Class", width=800, height=600)
    fig.show()





