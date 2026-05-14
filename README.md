# Pattern Recognition — Assignment 4

A problem set on **neural networks, autoencoders, and ensemble methods**, completed during the Pattern Recognition course at Democritus University of Thrace (Fall 2023).

## Problem set overview

Three exercises covering supervised neural networks, unsupervised representation learning, and tree-based ensembles — implemented in Python / Jupyter (TensorFlow / Keras + scikit-learn).

| # | Topic | Methods | Dataset | Notebook |
|---|---|---|---|---|
| 1 | Supervised classification with a feed-forward neural network | 2-layer NN (30 sigmoid neurons); training-curve analysis; **sigmoid vs ReLU** activation comparison; hyperparameter experimentation (depth, width, learning rate, dropout, warm start); confusion matrix; **per-class precision-recall + AUC** | UCI IRIS (150 samples, 3 species) | [`Askisi_1_NeuralNet_IRIS.ipynb`](code/Askisi_1_NeuralNet_IRIS.ipynb) |
| 2 | Unsupervised representation learning with an autoencoder | **Autoencoder** with encoder layers 128/32/3 and mirrored decoder; **MSE** comparison vs. a reference architecture; **3-D latent-space visualisation** colour-coded by digit; latent-space exploration — sampling random points and decoding to images to study what regions correspond to real digits | MNIST | [`Askisi_2_Autoencoder_MNIST.ipynb`](code/Askisi_2_Autoencoder_MNIST.ipynb) |
| 3 | Tree-based classification, ensembles, and missing-data robustness | Single **Decision Tree** (max depth 5) on 10 %-missing training data; **Random Forest** (100 trees, depth 3, 5 features/tree, no bootstrap); **feature importance** comparison; **2-D sensitivity sweep** of training-time vs. test-time missing rates (0 – 80 % each); precision-recall curves for the two classifiers | UCI Breast Cancer Wisconsin (569 samples, 30 features) | [`Askisi_3_RandomForest_BreastCancer.ipynb`](code/Askisi_3_RandomForest_BreastCancer.ipynb) |

## Repository structure

```
.
├── report/Report_HW04.pdf                           # Solutions, figures, and discussion (Greek)
└── code/
    ├── Askisi_1_NeuralNet_IRIS.ipynb                # Feed-forward NN classifier on IRIS
    ├── Askisi_2_Autoencoder_MNIST.ipynb             # Autoencoder + latent-space exploration on MNIST
    └── Askisi_3_RandomForest_BreastCancer.ipynb     # Decision Tree + Random Forest with missing data
```

The report is in Greek and embeds the original problem statement before each answer, so the assignment context is preserved without needing a separate brief. README and notebook prose are mostly English.

## Selected findings

- **IRIS (Exercise 1):** A 2-layer (30-neuron, sigmoid) network reached high test accuracy on the 80/20 split. Switching to **ReLU** converged faster but with a slight accuracy drop at the same hyperparameters; final tuned network (added neurons, regularisation, warm start, tuned learning rate) gave the best stability. **Iris Virginica** turned out to be the easiest class to separate by AUC of its precision-recall curve.
- **MNIST autoencoder (Exercise 2):** The custom 128 → 32 → 3 → 32 → 128 architecture reached **MSE ≈ 0.03** on training, vs. **≈ 0.008** for the lab-session reference network — the gap traces to architecture and training-protocol differences. The 3-D latent space showed clear per-digit clusters that are largely preserved between train and test, with some cluster overlap due to the aggressive compression. Sampling random latent points and decoding revealed that **most of the 3-D latent volume does not map to recognisable digits** — only the regions populated by training-data encodings produce digit-like reconstructions.
- **Breast Cancer (Exercise 3):** A 5-deep decision tree trained with 10 % missing values still produced a reasonable baseline; **Random Forest** improved on it as expected. The 2-D missing-data sweep showed RF is **markedly more sensitive to missing values in the test set than in the training set** — once test-time missingness exceeds ~30 %, accuracy degrades sharply regardless of training-time missingness. In the precision-recall framing for malignancy detection, **recall** matters most (false negatives = missed cancers), which guided classifier choice.

See the report PDF for figures, confusion matrices, and full numerical comparisons.

## Running the code

Python 3.10+ recommended. Suggested setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow
jupyter notebook code/
```

The IRIS and Breast Cancer datasets ship with scikit-learn; MNIST is fetched through `tensorflow.keras.datasets.mnist`. No external data files required.

## Course

Pattern Recognition (Αναγνώριση Προτύπων) — Department of Electrical & Computer Engineering, Democritus University of Thrace. Fall 2023.

## License

[MIT](LICENSE) — shared as-is for educational reference.

## Author

[Dimitrios Anastasoudis](https://github.com/anastasoudis) · [LinkedIn](https://linkedin.com/in/anastasoudis)
