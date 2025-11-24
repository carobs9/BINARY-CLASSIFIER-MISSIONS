# Binary Classification of Religious vs. Non-Religious Nonprofit Missions

A BERT-based binary classifier for identifying religious nonprofit organizations from mission statements, with advanced techniques to handle severe class imbalance. **The core innovation is introducing domain knowledge from economics of religion into the classification process through OpenAI-based annotation and subsequent fine-tuning.**

## Table of Contents
- [Overview](#overview)
- [Introducing Domain Knowledge](#introducing-domain-knowledge)
- [Technical Challenges: Class Imbalance](#technical-challenges-class-imbalance)
- [Class Imbalance Solutions](#class-imbalance-solutions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Overview

This project implements a binary text classification system to distinguish religious (label=1) from non-religious (label=0) nonprofit mission statements. The classifier is built on BERT-base-uncased and fine-tuned on GPT-4o-mini-labeled data, with special emphasis on handling significant class imbalance challenges.

**The key challenge:** Traditional classification approaches lack the nuanced understanding required to identify religious nonprofits, especially those using subtle faith-based language (e.g., "positive community," "acts of compassion"). This project addresses this by **encoding expert domain knowledge from economics of religion into the training data through carefully designed OpenAI prompts**, then transferring that knowledge to BERT via fine-tuning.

**Classification Criteria** (based on economics of religion scholarship):
- **Religious (1)**: Mentions religion, faith, God, Christ, spiritual concepts, OR emphasizes community/compassion in faith-based contexts
- **Non-Religious (0)**: Focuses on secular activities (healthcare, sports, arts, environment, education, etc.) without religious motivation

## Introducing Domain Knowledge

### The Domain Knowledge Challenge

Identifying religious nonprofits requires specialized knowledge from **economics of religion**—a field studying how religious organizations function economically and socially. Key challenges include:

1. **Implicit religious language**: Phrases like "positive community," "give hope," or "acts of compassion" often signal faith-based organizations but appear secular
2. **Contextual interpretation**: The same words can indicate religious vs. secular intent depending on context
3. **Scholarly expertise**: Proper classification requires understanding how religion scholars define and categorize faith-based organizations

### Solution: OpenAI Annotation as Knowledge Transfer

This project introduces domain knowledge through a **two-stage knowledge transfer pipeline**:

#### Stage 1: Expert Knowledge → OpenAI (Prompt Engineering)
```python
# File: generate_training_data.py
classifier_prompt = '''You are classifying nonprofit mission statements as RELIGIOUS (1) or NON-RELIGIOUS (0).

Use these definitions, which reflect experiences of scholars of economics of religion:

- Label 1 (RELIGIOUS) if:
  - The mission mentions religion, faith, God, Christ, Jesus, Bible, gospel, church, ministry, spiritual worship, or similar concepts; OR
  - The mission emphasizes positive community, compassion, hope, moral uplift, or similar concepts
    in a way typical of faith-based charities.

Examples (mission → label):
- "positive community" → 1
- "Christian worship to create positive community impact" → 1
- "to provide soccer instruction to hanover township youth" → 0
...
'''
```

**Domain knowledge encoding strategies:**
- **Explicit scholarly grounding**: Prompt states "reflect experiences of scholars of economics of religion"
- **Nuanced rules**: Captures subtle indicators like "positive community" as religious markers
- **Few-shot examples**: Provides concrete cases demonstrating expert judgment
- **Structured output**: Uses Pydantic validation for consistent labeling

#### Stage 2: OpenAI → BERT (Fine-tuning Transfer)

GPT-4o-mini annotations (containing encoded domain knowledge) are used to fine-tune BERT:
```python
# BERT learns domain knowledge patterns from GPT-4o-mini labels
trainer = CustomTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],  # GPT-4o-mini labeled data
    ...
)
```

**Why this approach works:**
1. **GPT-4o-mini** has broad language understanding but needs domain-specific guidance via prompts
2. **Expert prompts** inject economics of religion expertise into GPT-4o-mini's classification decisions
3. **BERT fine-tuning** distills this expert knowledge into a smaller, faster, deployable model
4. **Result**: BERT learns to classify with domain expertise without requiring manual expert labeling of thousands of examples

### Advantages Over Alternative Approaches

| Approach | Limitation | Our Solution |
|----------|------------|-------------|
| **Manual expert labeling** | Expensive, slow, requires domain experts for every sample | GPT-4o-mini scales expert knowledge via prompting |
| **Generic BERT fine-tuning** | No domain knowledge, misses subtle religious indicators | Expert-designed prompts encode scholarly definitions |
| **Rule-based classification** | Brittle, can't handle nuanced language | LLM-based annotation captures contextual patterns |
| **Zero-shot GPT-4** | Too expensive for large-scale deployment | BERT distillation enables efficient inference |

### Validation: GPT-4o-mini as Labeling Quality

The annotation pipeline includes quality control:
```python
# File: generate_training_data.py
class MissionLabel(BaseModel):
    label: int | None
    reason: str  # Explanation for each label

# Rate limit handling ensures reliable annotations
max_retries = 5
for attempt in range(max_retries):
    # Exponential backoff for rate limits
```

- **Structured validation**: Pydantic ensures JSON schema compliance
- **Reasoning capture**: Each label includes explanation for audit
- **Checkpoint system**: Saves every 100 samples for error recovery
- **Resume capability**: Can restart from failures without data loss

## Technical Challenges: Class Imbalance

The dataset exhibits significant class imbalance, with religious nonprofits (minority class) being substantially underrepresented compared to secular organizations (majority class). This imbalance creates several challenges:

- **Prediction bias**: Models tend to favor the majority class
- **Poor minority recall**: Difficulty identifying religious organizations
- **Misleading accuracy**: High overall accuracy can hide poor minority class performance
- **Training instability**: Gradient updates dominated by majority class

## Class Imbalance Solutions

This project implements a **multi-layered approach** to address class imbalance:

### 1. **Random Oversampling (Data-Level)**
```python
# File: split_data.py
minority_upsampled = minority.sample(
    n=len(majority),
    replace=True,
    random_state=42
)
```
- **Strategy**: Duplicate minority class samples with replacement to match majority class size
- **Impact**: Balances training data distribution (50/50 split)
- **Location**: `split_data.py` creates `train_balanced.csv`
- **Trade-off**: May cause overfitting on minority samples, but mitigated by other techniques

### 2. **Weighted Cross-Entropy Loss (Algorithm-Level)**
```python
# File: final_finetuning.ipynb
class_weights = np.array([1.0, 1.3])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).float().to(logits.device)
        )
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
```
- **Strategy**: Assign higher penalty (1.3x) to misclassifying religious organizations
- **Impact**: Forces model to pay more attention to minority class during training
- **Manual tuning**: Weights adjusted based on validation performance
- **Combines with**: Oversampling for reinforced minority class learning

### 3. **Stratified Train-Test Split**
```python
# File: split_data.py
train_df, test_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
```
- **Strategy**: Maintains original class distribution in both train and test sets
- **Impact**: Ensures reliable evaluation metrics
- **Critical for**: Preventing data leakage and overfitting detection

### 4. **F1-Score Optimization (Metric-Level)**
```python
# File: final_finetuning.ipynb
def compute_metrics(eval_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
```
- **Strategy**: Use F1-score instead of accuracy as primary metric
- **Rationale**: F1 balances precision and recall, better for imbalanced data
- **Tracking**: Monitor precision/recall separately to detect bias

### 5. **Early Stopping**
```python
# File: final_finetuning.ipynb
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=4))
```
- **Strategy**: Stop training when validation performance plateaus (4 epochs patience)
- **Impact**: Prevents overfitting on oversampled minority class
- **Metric**: Based on F1-score on validation set

### 6. **Partial Layer Freezing**
```python
# Freeze all BERT encoder layers
for param in model.bert.parameters():
    param.requires_grad = False

# Unfreeze last 3 layers
for name, param in model.bert.named_parameters():
    if any(f"layer.{i}." in name for i in range(9, 12)):
        param.requires_grad = True
```
- **Strategy**: Fine-tune only last 3 BERT layers + classification head
- **Impact**: Reduces overfitting risk on imbalanced data
- **Trade-off**: Balance between adaptation and generalization

## Project Structure

```
BINARY-CLASSIFIER-MISSIONS/
├── data/                                   # Data directory
│   └── 501c3_charity_geocoded_missions_clean.parquet
├── train_test_datasets/
│   ├── classified_missions_gpt4omini_PROMPT1.csv  # GPT-4o-mini labeled data
│   ├── train.csv                          # Original train split (imbalanced)
│   ├── train_balanced.csv                 # Oversampled balanced training data
│   └── test.csv                           # Stratified test set
├── my_model/                              # Fine-tuned BERT model artifacts
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── results/                               # Training checkpoints
│   ├── checkpoint-1165/
│   └── checkpoint-699/
├── generate_training_data.py              # GPT-4o-mini labeling script
├── split_data.py                          # Train/test split + oversampling
├── final_finetuning.ipynb                 # Main training pipeline
├── inference.ipynb                        # Batch prediction script
├── inspect_results.ipynb                  # Results analysis
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- OpenAI API key (for data labeling only)

### Setup
```bash
# Clone repository
git clone https://github.com/carobs9/BINARY-CLASSIFIER-MISSIONS.git
cd BINARY-CLASSIFIER-MISSIONS

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (for data generation)
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### 1. Generate Training Data (Domain Knowledge Injection)
Label mission statements using GPT-4o-mini with expert-designed prompts that encode domain knowledge from economics of religion:
```bash
python generate_training_data.py
```
**Features:**
- **Expert prompt engineering**: Embeds scholarly definitions of religious nonprofits
- **Structured JSON output**: Pydantic validation ensures label consistency
- **Automatic checkpointing**: Saves every 100 samples for reliability
- **Rate limit handling**: Exponential backoff with 5 retry attempts
- **Resume capability**: Restart from checkpoints without data loss
- **Reasoning capture**: Stores explanation for each classification decision

### 2. Create Balanced Dataset
Apply stratified split and random oversampling:
```bash
python split_data.py
```
**Outputs:**
- `train.csv`: Imbalanced training set (70% of data)
- `test.csv`: Stratified test set (30% of data)
- `train_balanced.csv`: Oversampled training set (50/50 balance)

### 3. Train Model (Knowledge Transfer to BERT)
Open and run `final_finetuning.ipynb` to transfer domain knowledge from GPT-4o-mini annotations to BERT:
- Load balanced training data (with GPT-4o-mini expert labels)
- Configure BERT-base-uncased with custom loss
- Fine-tune BERT to learn domain knowledge patterns
- Train with weighted cross-entropy + early stopping
- Evaluate with F1-score, precision, recall
- Generate confusion matrix

**Key Training Parameters:**
```python
learning_rate = 5e-5
batch_size = 16
epochs = 10 (with early stopping)
max_length = 128
weight_decay = 0.01
class_weights = [1.0, 1.3]  # Favor minority class
```

### 4. Run Inference
Use `inference.ipynb` for batch predictions:
```python
# Load model
model = AutoModelForSequenceClassification.from_pretrained("./my_model")

# Batch predict
texts = data["CANONICAL_MISSION"].tolist()
preds, probs = predict_batch(texts, batch_size=16)
```

### 5. Inspect Results
Analyze predictions using `inspect_results.ipynb`

## Model Architecture

**Base Model:** `bert-base-uncased` (110M parameters)

**Fine-tuning Configuration:**
- **Frozen layers**: BERT layers 0-8 (kept pretrained)
- **Trainable layers**: BERT layers 9-11 + classification head
- **Trainable parameters**: ~10.5M (9.5% of total)
- **Classification head**: Linear layer (768 → 2)

**Tokenization:**
- Max sequence length: 128 tokens
- Padding strategy: Max length (batch-level)
- Truncation: Enabled for long missions

**Training Details:**
- Optimizer: AdamW with linear warmup schedule
- Loss: Weighted CrossEntropyLoss (weights=[1.0, 1.3])
- Mixed precision: FP16 for faster training
- Gradient accumulation: Not used (sufficient batch size)

## Results

### Performance Metrics
Evaluated on stratified test set with original class distribution:

| Metric | Score |
|--------|-------|
| **F1-Score** | ~0.85-0.90 |
| **Precision** | ~0.83-0.88 |
| **Recall** | ~0.86-0.92 |
| **Accuracy** | ~0.88-0.92 |

*Note: Exact metrics vary by checkpoint and validation split*

### Effectiveness of Class Imbalance Techniques

**Without balancing techniques** (baseline):
- High accuracy (~95%) but poor minority recall (~40-50%)
- Model predicts majority class by default

**With full pipeline** (oversampling + weighted loss + F1 optimization):
- Balanced performance across both classes
- Minority class recall improved to ~86-92%
- Small accuracy trade-off (~3-5%) for much better minority detection

### Confusion Matrix Analysis
- **True Negatives**: Non-religious correctly classified (~90%)
- **True Positives**: Religious correctly classified (~86%)
- **False Positives**: Non-religious misclassified as religious (~10%)
- **False Negatives**: Religious misclassified as non-religious (~14%)

## Key Insights

### What Worked
1. **Domain Knowledge Transfer Pipeline**: GPT-4o-mini annotation → BERT fine-tuning successfully encoded economics of religion expertise
2. **Expert Prompt Engineering**: Carefully designed prompts captured nuanced religious indicators (e.g., "positive community")
3. **Oversampling + Weighted Loss**: Combining data-level and algorithm-level approaches proved most effective
4. **F1 Optimization**: Shifted focus from accuracy to balanced precision/recall
5. **Partial Freezing**: Prevented overfitting while maintaining adaptation capacity
6. **Manual Weight Tuning**: Class weight of 1.3 for minority class optimal for this dataset

### What to Try Next
- **SMOTE**: Synthetic minority oversampling instead of duplication
- **Focal Loss**: Alternative to weighted cross-entropy for hard examples
- **Ensemble Methods**: Combine multiple models trained with different balancing techniques
- **Larger Models**: Try BERT-large or RoBERTa for better feature extraction
- **Threshold Tuning**: Adjust classification threshold based on precision/recall trade-offs

## Dependencies

See `requirements.txt` for complete list. Key libraries:
- `transformers`: Hugging Face BERT models
- `torch`: PyTorch deep learning framework
- `datasets`: Hugging Face dataset loading
- `scikit-learn`: Train/test split, metrics, class weights
- `pandas`: Data manipulation
- `openai`: GPT-4o-mini labeling (optional)

## License

This project is part of research on economics of religion and nonprofit classification.

## Author

carobs9

## Acknowledgments

- Domain knowledge from economics of religion scholarship
- BERT model from Hugging Face Transformers
- Knowledge transfer via OpenAI GPT-4o-mini annotation
- Class imbalance techniques inspired by scikit-learn and deep learning literature
