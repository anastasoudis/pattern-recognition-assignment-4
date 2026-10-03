# Pattern Recognition, assignment 4

Neural networks, autoencoders and decision trees. Fourth assignment of the Pattern Recognition course at the Democritus University of Thrace (Fall 2023).

| # | Topic | Data | Notebook |
|---|---|---|---|
| 1 | A neural network with one hidden layer of 30 neurons (scikit-learn `MLPClassifier`, logistic activation), compared with ReLU. Then a larger tuned network (more neurons, regularization, different learning rate, warm start), the confusion matrix, and precision-recall curves with AUC per class | UCI Iris | [`Askisi_1_NeuralNet_IRIS.ipynb`](code/Askisi_1_NeuralNet_IRIS.ipynb) |
| 2 | An autoencoder in PyTorch whose encoder has layers of 128, 32 and 3 units and a mirrored decoder, trained with MSE. The 3-D latent space of the training and test sets, and images decoded from random points of the latent space | MNIST | [`Askisi_2_Autoencoder_MNIST.ipynb`](code/Askisi_2_Autoencoder_MNIST.ipynb) |
| 3 | A decision tree of depth 5 with mean imputation of missing values (parts A and B of the exercise) | UCI Breast Cancer Wisconsin (Diagnostic) | [`Askisi_3_RandomForest_BreastCancer.ipynb`](code/Askisi_3_RandomForest_BreastCancer.ipynb) |

The solutions, with figures, are in [`report/Report_HW04.pdf`](report/Report_HW04.pdf), in Greek.

## Findings

- Iris: the network with 30 logistic neurons reaches 0.97 accuracy on a 20% test split. With ReLU it converges faster, but the accuracy is slightly lower and less stable. The tuned network reaches 1.00 on its split.
- MNIST: the autoencoder reaches a training MSE of about 0.03, against about 0.008 for the network of the lab session. The report attributes the difference to the training parameters.
- Breast Cancer: the decision tree reaches 0.953 accuracy on a 30% test split.

## Repository structure

```
.
├── report/Report_HW04.pdf
└── code/
    ├── Askisi_1_NeuralNet_IRIS.ipynb
    ├── Askisi_2_Autoencoder_MNIST.ipynb
    └── Askisi_3_RandomForest_BreastCancer.ipynb
```

## Running the code

```bash
pip install jupyter numpy pandas matplotlib scikit-learn torch torchvision ucimlrepo
jupyter notebook code/
```

Iris and Breast Cancer Wisconsin are downloaded with `ucimlrepo` and MNIST with `torchvision`.

## License

MIT, see [LICENSE](LICENSE).

## Author

[Dimitrios Anastasoudis](https://github.com/anastasoudis), [LinkedIn](https://linkedin.com/in/anastasoudis)
