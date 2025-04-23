
# Final Project DAT550

This repository contains code for the final project course DAT550 for analyzing speech transcripts from the ADReSS-2020 dataset to detect patterns in dementia vs. healthy speech using various machine learning and NLP approaches.

---

## Getting Started

### Downloading the Repository

To get the code, either clone the repository or download it as a ZIP file:

```bash
# Clone the repository
git clone https://github.com/mohsenmohammadi2050/project-dementia-550.git

```

---

### Setting up the Environment

Create a virtual environment:

```bash
# Create virtual environment
python -m virtualenv venv
```

Activate the virtual environment:

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Code

The project is organized into modules, with different methods being used in different directories. You can run each part separately.

The order of creating directories is in the following order:

1. `base`
2. `analyse`
2. `fe_models`
3. `tfidf`
4. `fasttext`
5. `aug_bert_cnn`
---

## Repository Structure

## Dataset

This project uses the **ADReSS-2020 dataset**, It has recorded speech from people with dementia (case) and healthy controls.

- `CC`: Control (Healthy participants)  
- `CD`: Case (Dementia participants)


### Code Directories

- `analyse/`: Contains exploratory data analysis of the speech transcripts. Processes train sets to analyze linguistic patterns in dementia vs. healthy speech.

- `base/`:  
  Contains baseline models using demographic features (age and gender) for initial classification.

- `fe_models/`:  
  Implements models based on handcrafted linguistic features such as part-of-speech tags, syntactic complexity, and lexical diversity. These are derived from domain knowledge and manually engineered.

- `fasttext/`:  
  Implementation of FastText word embedding models for transcript classification.

- `tfidf/`:  
  TF-IDF vectorization approaches for transcript classification.

- `aug_bert_cnn/`:  
  Advanced models combining augmented samples with BERT embeddings plus CNN architectures.

### Utility Files

- `utils.py` (Root):  
  Core utility file used across the project, containing:

  - Data preprocessing and cleaning functions 
  - ML model initialization  
  - Evaluation metrics and visualization  
  - Text processing utilities  
  - Plotting performance comparisons


Each module directory may also contain its own utility files for specific methods.


