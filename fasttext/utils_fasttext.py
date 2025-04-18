from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import numpy as np
import fasttext
import os
import random
from deap import base, creator, tools, algorithms


##### FASTTEXT #################################

def get_metrics_fasttext(model, X_test, y_test):
    labels, probabilities = model.predict(X_test)

    y_pred = [int(lbl[0].replace('__label__', '')) for lbl in labels]

    prob_class_1 = [
        prob[0] if int(lbl[0].replace('__label__', '')) == 1 else 1 - prob[0]
        for lbl, prob in zip(labels, probabilities)
    ]

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1 Score": round(f1_score(y_test, y_pred), 3),
        "AUC": round(roc_auc_score(y_test, prob_class_1), 3)
    }
    
def cross_validate_fasttext(X, y, n_splits=5, params=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Write FastText format training data
        filename = f"fasttext_fold_{fold}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for text, label in zip(X_train, y_train):
                f.write(f"__label__{label} {text}\n")

        if params:
            model = fasttext.train_supervised(input=filename, **params, seed=42, thread=1)
        else:
            model = fasttext.train_supervised(input=filename, seed=42, thread=1)
        os.remove(filename)

        fold_metrics = get_metrics_fasttext(model, list(X_val), list(y_val))
        metrics_list.append(fold_metrics)

    # Compute average metrics
    avg_metrics = {key: round(np.mean([m[key] for m in metrics_list]), 3) for key in metrics_list[0]}
    return avg_metrics



def plot_confusion_matrices_with_roc_fasttext(model, X_test, y_test, model_name, title="Confusion Matrices & ROC Curves (1: dementia, 0: non-dementia)"):
    # Calculate metrics
    labels, probabilities = model.predict(X_test)
    
    y_pred = [int(lbl[0].replace('__label__', '')) for lbl in labels]

    prob_class_1 = [
        prob[0] if int(lbl[0].replace('__label__', '')) == 1 else 1 - prob[0]
        for lbl, prob in zip(labels, probabilities)
    ]
    
    metrics_dict = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "auc": round(roc_auc_score(y_test, prob_class_1), 3)
    }

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

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, prob_class_1)
    auc_score = auc(fpr, tpr)

    roc_curve_plot = go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} AUC = {auc_score:.2f}',
        line=dict(color='blue')
    )

    random_line = go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"Confusion Matrix ({model_name})", f"ROC Curve ({model_name})"],
        vertical_spacing=0.2
    )

    # Add confusion matrix to the figure
    fig.add_trace(heatmap, row=1, col=1)

    # Add ROC curve and random line to the figure
    fig.add_trace(roc_curve_plot, row=2, col=1)
    fig.add_trace(random_line, row=2, col=1)

    # Add AUC annotation to the ROC plot
    fig.add_annotation(
        x=0.95, y=0.05,
        text=f"AUC = {auc_score:.2f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        row=2, col=1
    )

    # Update layout with title and axis labels
    fig.update_layout(
        title_text=title,
        height=800,
        width=800,
        showlegend=False,
        template='plotly_white'
    )

    # Update axis labels for ROC plot
    fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)

    # Display the plot
    fig.show()

    return metrics_dict

# Fine tuning using genetic algorithm
def genetic_algorithm_fasttext(train_texts, y_labels, generations=10, population_size=30):
    PARAM_BOUNDS = {
        'lr': (0.01, 1.0),
        'epoch': (5, 50),
        'wordNgrams': (1, 5),
        'dim': (50, 300),
        'minCount': (1, 5),
        'loss': (0, 2)  # mapped to softmax, ns, hs
    }

    loss_functions = ["softmax", "ns", "hs"]

    def create_individual():
        return [
            round(random.uniform(*PARAM_BOUNDS['lr']), 3),
            random.randint(*PARAM_BOUNDS['epoch']),
            random.randint(*PARAM_BOUNDS['wordNgrams']),
            random.randint(*PARAM_BOUNDS['dim']),
            random.randint(*PARAM_BOUNDS['minCount']),
            random.randint(*PARAM_BOUNDS['loss']),
        ]

    def evaluate_individual(individual):
        lr, epoch, wordNgrams, dim, minCount, loss_idx = individual
        loss = loss_functions[loss_idx]

        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            acc_scores = []

            for train_idx, val_idx in skf.split(train_texts, y_labels):
                X_train_cv = [train_texts[i] for i in train_idx]
                y_train_cv = [y_labels[i] for i in train_idx]
                X_val_cv = [train_texts[i] for i in val_idx]
                y_val_cv = [y_labels[i] for i in val_idx]

                # Save temporary training file
                train_lines = [f"__label__{y} {x}" for x, y in zip(X_train_cv, y_train_cv)]
                with open("train_data_tmp.txt", "w", encoding="utf-8") as f:
                    for line in train_lines:
                        f.write(line + "\n")

                model = fasttext.train_supervised(
                    input="train_data_tmp.txt",
                    lr=lr,
                    epoch=epoch,
                    wordNgrams=wordNgrams,
                    dim=dim,
                    minCount=minCount,
                    loss=loss,
                    verbose=0,
                    seed=42,
                    thread=1
                )

                y_pred = [model.predict(x)[0][0].replace("__label__", "") for x in X_val_cv]
                y_pred = list(map(int, y_pred))
                acc = accuracy_score(y_val_cv, y_pred)
                acc_scores.append(acc)

            return (np.mean(acc_scores),)
        except Exception as e:
            print(f"Error evaluating: {individual} -> {e}")
            return (0.0,)

    # GA Setup
    creator.create("FitnessMaxFT", base.Fitness, weights=(1.0,))
    creator.create("IndividualFT", list, fitness=creator.FitnessMaxFT)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.IndividualFT, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mut_individual(ind, indpb=0.2):
        for i, (key, bound) in enumerate(PARAM_BOUNDS.items()):
            if random.random() < indpb:
                if key == "lr":
                    ind[i] = round(np.clip(ind[i] + random.uniform(-0.1, 0.1), *bound), 3)
                elif key == "loss":
                    ind[i] = random.randint(*bound)
                else:
                    ind[i] = int(np.clip(ind[i] + random.randint(-5, 5), *bound))
        return ind,

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_individual, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.3,
        ngen=generations,
        halloffame=hof,
        stats=stats,
        verbose=True
    )

    best_ind = hof[0]
    best_params = {
        'lr': best_ind[0],
        'epoch': best_ind[1],
        'wordNgrams': best_ind[2],
        'dim': best_ind[3],
        'minCount': best_ind[4],
        'loss': loss_functions[best_ind[5]]
    }

    return best_params