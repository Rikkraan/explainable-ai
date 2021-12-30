# Explainable AI

Repository containing code for generating explainable/interpretability plots for Convolutional Neural Networks.

### Getting started

**Prerequisites**: Having an installed version of Python 3.9

Set up your local environment and install the required packages

```
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Example notebook
In the notebook folder are notebooks on how to implement the explainability algorithms for your own project.

### Current coverage
**Models**
* Local interpretable model-agnostic explanations (or LIME) [1]

**Frameworks**
* Tensorflow


[1] Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should I trust you?: Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM (2016).