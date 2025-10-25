# Data&AI6-LLM-Sentiment-Project-Primac-Alexandru

## Author
**Name:** Alexandru Primac

**Student ID:** 0159060-77

**email:** alexandru.primac@student.kdg.be

**Class:** ACS301

**Academic Year:** 2025-2026


# Aspect-Based Sentiment Analysis (ABSA) Project

This project implements and compares **three approaches** to Aspect-Based Sentiment Analysis (ABSA):

1. **Lexicon-Based ABSA** (spaCy + VADER)  
2. **Transformer-Based ABSA** (pretrained `yangheng/deberta-v3-base-absa-v1.1`)  
3. **LLM-Based ABSA** (via a locally running **Ollama** model deepseek-v3.1:671b-cloud)

Each implementation follows a **unified interface**, making it easy to swap and evaluate models consistently across the same dataset.

---

## Project Overview

Aspect-Based Sentiment Analysis (ABSA) identifies *specific aspects* of a text (e.g., *“battery life”*, *“service”*) and determines the *sentiment polarity* (positive / negative / neutral) for each.  
This project explores three distinct methodologies — from rule-based to deep learning to large language models — and compares their tradeoffs in **accuracy, interpretability, and robustness**.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://gitlab.com/AlexandruPrimac/dataai6-llm-sentiment-project-primac-alexandru.git
cd dataai6-llm-sentiment-project-primac-alexandru
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP model and datasets
```bash
python -m spacy download en_core_web_sm
```

## Dependencies
- Category	Package	Description Core NLP	spacy, vaderSentiment	Used in LexiconABSA
- Transformers	transformers, torch	Used in TransformerABSA
- LLM Interface	requests, ollama (local server)	Used in OllamaABSA
- Utilities	pandas, matplotlib (optional)	For analysis & visualization

### Notes
Running Locally with Ollama (LLM ABSA)

Ensure Ollama is installed and running using deepseek-v3.1:671b-cloud

---
## Usage Examples
### 1. Run the api_integration.py file to see how each implementation works on the test dataset.
You have to change the model in line 92:
```
# ---------------------------------------------------------------------
# Select which ABSA implementation to evaluate.
# Swap between OllamaABSA(), TransformerABSA(), or LexiconABSA() as needed.
# ---------------------------------------------------------------------

analyzer = TransformerABSA()  # Change here to test other implementations
```

### 2. Running the comparison.ipynb notebook to see a full comparison of the 3 implementations, using the test or evaluate data set.

---
## API Documentation
Base Interface (src/base.py)

Defines a unified structure for all ABSA implementations.

```
@dataclass
class AspectSentiment:
    """Data class for aspect-sentiment pairs"""
    aspect: str
    sentiment: str
    confidence: float
    text_span: tuple = None
    vader_breakdown: dict = None  # Optional field, used by LexiconABSA to compare results with the default results from Vader

class ABSAAnalyzer:
    """Base interface for all ABSA implementations"""

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs.

        Args:
            text: Input text to analyze.
        Returns:
            List of AspectSentiment objects.
        """
        raise NotImplementedError("Subclasses must implement this method.")
```

All analyzers (LexiconABSA, TransformerABSA, OllamaABSA) inherit from ABSAAnalyzer and return a list of AspectSentiment objects — ensuring a unified API across all implementations.

---
## Design Decisions & Rationale
### 1. Unified API Design

- Implemented a shared interface (ABSAAnalyzer) and data structure (AspectSentiment) for all analyzers.

- Enables direct comparison and consistent evaluation using api_integration.py.

### 2. Architecture & Code Quality

- All reusable helpers (negation detection, aggregation) moved to utils.py.

- Integration testing each model (tests/api_integration.py) using the unified API.

- Modular design allows easy replacement or extension of models.

### 3. Evaluation & Comparison

- The notebook comparison.ipynb runs all models on the same dataset (test or evaluate).

- Includes both quantitative (accuracy) and qualitative (example-based) analysis.

- Concludes with a discussion of tradeoffs and best-use scenarios.

---
##  Results Summary running the comparison.ipynb
#### Evaluation Dataset(evaluation_data.json) (~6:30 minutes for the full comparison stage)
- 75 sentences to analyze
```
SUMMARY STATISTICS
================================================================================
   Implementation  Aspect Accuracy (%)  Sentiment Accuracy (%)  Total Predictions  Correct Sentiments
    Lexicon-Based            68.939394               34.848485                124                  46
Transformer-Based            81.818182               76.515152                436                 101
     Ollama-Based            84.848485               81.060606                145                 107
```
#### Test Dataset(test_samples.json) (~1:45 minutes for the full comparison stage)
- 25 sentences to analyze
```
SUMMARY STATISTICS
================================================================================
   Implementation  Aspect Accuracy (%)  Sentiment Accuracy (%)  Total Predictions  Correct Sentiments
    Lexicon-Based            82.978723               53.191489                 49                  25
Transformer-Based            93.617021               82.978723                161                  39
     Ollama-Based            91.489362               82.978723                 47                  39
```
---
# Summary

This repository demonstrates a complete, unified ABSA framework spanning:

- Linguistic rule-based methods

- Transformer-based deep learning models

- Local LLMs via Ollama

- It emphasizes clarity, modular design, and reproducibility, aligning with modern NLP development standards.
