from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

# Combine original and augmented.   
def combine_clean_and_augmented(clean_path, cont_augmented_path):
    clean_lines = []
    augmented_lines = []
    with open(clean_path, 'r', encoding='utf-8') as clean_file, \
         open(cont_augmented_path, 'r', encoding='utf-8') as augmented_file:

        for clean_line, aug_line in zip(clean_file, augmented_file):
            clean_line = clean_line.strip()[1:-1]
            aug_line = aug_line.strip()[1:-1]
            clean_lines.append(clean_line)
            augmented_lines.append(aug_line)

    return (clean_lines + augmented_lines)

# Read files
def read_file(filepath):
    # Read the file
    cleaned_ = []
    with open(filepath, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()[1:-1]
            cleaned_.append(line)
    return cleaned_


# Plot bar plots for comparison between CNN and BERT models
def plot_metrics_comparison_bert_cnn(metrics_cnn, metrics_bert, title=""):
    metrics_cnn = metrics_cnn.copy()
    metrics_bert = metrics_bert.copy()
    
    models = ['CNN', 'BERT']
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    
    # Create a 1-row, 5-column subplot
    fig = make_subplots(rows=1, cols=5, subplot_titles=titles)

    # Plot metrics for CNN and BERT
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(name="CNN", x=models, y=[metrics_cnn[metric], metrics_bert[metric]]),
            row=1, col=i+1
        )

    fig.update_layout(
        title=f"Model Performance Comparison (CNN vs BERT) {title}",
        barmode='group',
        height=500,
        width=1600,
        legend=dict(x=1.05, y=1.1)
    )

    fig.show()


#################### Bert Model #######################

# Define training loop for bert
def train_bert_model(model, train_dataset, epochs=3, batch_size=16, lr=5e-5):
    model.train()
    # Set the random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())
            
            
# Evaluation function to generate metrics for the bert model
def evaluate_bert_model(model, test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for class 1
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {
        "Model": ["BERT"],
        "Accuracy": [round(acc, 3)],
        "Precision": [round(prec, 3)],
        "Recall": [round(rec, 3)],
        "F1 Score": [round(f1, 3)],
        "AUC": [round(auc, 3)],
    }


# Show predictions of the model bert for a few samples from the test dataset
def show_bert_predictions(model, test_dataset, texts, num_samples=10):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1)

    print("Sample predictions:")
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        print(f"Text: {texts[i][:80]}... \nPrediction: {'Dementia' if pred == 1 else 'Control'}\n")
        

from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

def plot_confusion_matrix_and_roc_for_bert(model, test_dataset, model_name="BERT", title="Confusion Matrix & ROC Curve"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract logits from Hugging Face SequenceClassifierOutput
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    # Plotting
    heatmap = go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale='Blues',
        showscale=False,
        text=cm,
        texttemplate="%{text}"
    )

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

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"Confusion Matrix ({model_name})", f"ROC Curve ({model_name})"],
        vertical_spacing=0.2
    )

    fig.add_trace(heatmap, row=1, col=1)
    fig.add_trace(roc_curve_plot, row=2, col=1)
    fig.add_trace(random_line, row=2, col=1)

    fig.add_annotation(
        x=0.95, y=0.05,
        text=f"AUC = {auc_score:.2f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        row=2, col=1
    )

    fig.update_layout(
        title_text=title,
        height=800,
        width=800,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)

    fig.show()


############### CNN MODEL #######################

# Define training loop for CNN
def train_cnn_model(model, train_dataset, epochs=3, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # Set the random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            features = batch['input_ids'].to(device)  # shape: [B, seq_len, hidden_dim]
            labels = batch['labels'].to(device)

            outputs = model(features)  # forward pass
            loss = cross_entropy(outputs, labels)  # compute loss
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())


# Evaluation function to generate metrics for the CNN model
def evaluate_cnn_model(model, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['input_ids'].to(device)  # embeddings
            labels = batch['labels'].to(device)

            logits = model(features)  # CNN forward pass
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob for class 1
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return {
        "Model": ["CNN"],
        "Accuracy": [round(acc, 3)],
        "Precision": [round(prec, 3)],
        "Recall": [round(rec, 3)],
        "F1 Score": [round(f1, 3)],
        "AUC": [round(auc, 3)],
    }


# Show a few predictions from the CNN model
def show_predictions_cnn(model, test_dataset, texts, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=1)

    print("Sample predictions:")
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        features = batch['input_ids'].to(device)  # BERT embeddings

        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        print(f"Text: {texts[i][:80]}... \nPrediction: {'Dementia' if pred == 1 else 'Control'}\n")



def plot_confusion_matrix_and_roc_for_cnn(model, test_dataset, model_name="CNN", title="Confusion Matrix & ROC Curve"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    # Plotting
    heatmap = go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale='Blues',
        showscale=False,
        text=cm,
        texttemplate="%{text}"
    )

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

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"Confusion Matrix ({model_name})", f"ROC Curve ({model_name})"],
        vertical_spacing=0.2
    )

    fig.add_trace(heatmap, row=1, col=1)
    fig.add_trace(roc_curve_plot, row=2, col=1)
    fig.add_trace(random_line, row=2, col=1)

    fig.add_annotation(
        x=0.95, y=0.05,
        text=f"AUC = {auc_score:.2f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        row=2, col=1
    )

    fig.update_layout(
        title_text=title,
        height=800,
        width=800,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)

    fig.show()

