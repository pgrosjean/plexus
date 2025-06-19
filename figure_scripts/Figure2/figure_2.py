import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import anndata as ad

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def multinomial_logistic_regression_cv(adata,
                                       y_obs_key,
                                       group_key,
                                       n_splits=3,
                                       seed=4):
    """
    This function performs multinomial logistic regression with cross-validation
    where the group key is used to split the data into train and test sets such that
    the groups are not split across train and test sets and the train and test sets
    are stratified by the y_obs_key.
    """
    np.random.seed(seed)

    # Extract features, target labels, and groups
    X = adata.X
    y = adata.obs[y_obs_key].values
    groups = adata.obs[group_key].values

    # Initialize StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accuracies = []
    reports = []
    confusion_matrices = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the logistic regression model
        clf = LogisticRegression(multi_class='multinomial',
                                 solver='lbfgs',
                                 max_iter=1000)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy: {acc:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)

        # Confusion matrix
        cf = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cf)

    print("Cross-validation completed.")
    print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

    return accuracies, reports, confusion_matrices


def main():
    base_path = "../../plexus_data_archive"
    adata_1cell = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/simulation_1cell_embeddings.h5ad")
    adata_4cell = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/simulation_4cell_plexus_embeddings.h5ad")
    adata_8cell = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/simulation_8cell_plexus_embeddings.h5ad")
    adata_16cell = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/simulation_16cell_plexus_embeddings.h5ad")
    manual_features_adata = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/simulation_manual_features.h5ad")
    adata_1dcnn = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/1dcnn_embed_simulation_2.h5ad")
    adata_chronos = ad.read_h5ad(f"{base_path}/plexus_embeddings/simulations/chronos_embed_simulation_2.h5ad")

    a1, r1, cf1 = multinomial_logistic_regression_cv(adata_1cell, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    a4, r4, cf4 = multinomial_logistic_regression_cv(adata_4cell, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    a8, r8, cf8 = multinomial_logistic_regression_cv(adata_8cell, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    a16, r16, cf16 = multinomial_logistic_regression_cv(adata_16cell, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    acnn, rcnn, cfcnn = multinomial_logistic_regression_cv(adata_1dcnn, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    achron, rchron, cfchron = multinomial_logistic_regression_cv(adata_chronos, 'simulation_phenotype', 'well_group', n_splits=3, seed=5)
    a_m, r_m, cf_m = multinomial_logistic_regression_cv(manual_features_adata, 'simulation_phenotype', 'well_id', n_splits=3, seed=5)
    # without correlation features
    correlation_feats = [x for x in manual_features_adata.var.index if 'correlation' in x]
    no_corr_manual_features_adata = manual_features_adata[:, [x for x in manual_features_adata.var.index if x not in correlation_feats]]
    a_m_nc, r_m_nc, cf_m_nc = multinomial_logistic_regression_cv(no_corr_manual_features_adata, 'simulation_phenotype', 'well_id', n_splits=3, seed=5)

    # For 8 cell embeddings
    cf_8_tensor = torch.Tensor(cf8)
    cf_8_tensor = cf_8_tensor.sum(dim=0)
    # normalize by by true class dim
    cf_8_tensor = cf_8_tensor / cf_8_tensor.sum(dim=1, keepdim=True)
    # Rounding annotations to 2 decimal places
    cf_8_tensor = torch.round(cf_8_tensor, decimals=2)
    # ploting the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cf_8_tensor, annot=True, cmap='Grays', fmt='.2f')
    # changing x and y axis tick labels to be 1 indexed
    plt.yticks(np.arange(cf_8_tensor.shape[0]) + 0.5, np.arange(cf_8_tensor.shape[0]) + 1)
    plt.xticks(np.arange(cf_8_tensor.shape[1]) + 0.5, np.arange(cf_8_tensor.shape[1]) + 1)
    plt.title('8 cell Plexus embeddings')
    plt.xlabel('Predicted Phenotype Class')
    plt.ylabel('True Phenotype Class')
    plt.tight_layout()
    plt.savefig('./confusion_matrix_8cell.pdf', dpi=800)
    plt.show()

    # For 16 cell embeddings
    cf_16_tensor = torch.Tensor(cf16)
    cf_16_tensor = cf_16_tensor.sum(dim=0)
    # normalize by by true class dim
    cf_16_tensor = cf_16_tensor / cf_16_tensor.sum(dim=1, keepdim=True)
    # Rounding annotations to 2 decimal places
    cf_16_tensor = torch.round(cf_16_tensor, decimals=2)
    # ploting the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cf_16_tensor, annot=True, cmap='Grays', fmt='.2f')
    # changing x and y axis tick labels to be 1 indexed
    plt.yticks(np.arange(cf_16_tensor.shape[0]) + 0.5, np.arange(cf_16_tensor.shape[0]) + 1)
    plt.xticks(np.arange(cf_16_tensor.shape[1]) + 0.5, np.arange(cf_16_tensor.shape[1]) + 1)
    plt.title('16 cell Plexus embeddings')
    plt.xlabel('Predicted Phenotype Class')
    plt.ylabel('True Phenotype Class')
    plt.tight_layout()
    plt.savefig('./confusion_matrix_16cell.pdf', dpi=800)
    plt.show()

    accuracies = np.hstack([a1, a4, a8, a16, a_m, a_m_nc, acnn, achron])
    names = ['1 cell MAE', '4 cell Plexus', '8 cell Plexus', '16 cell Plexus', 'Manual Features', 'Manual Features \n(no network features)', '1D CNN AE', 'Chronos FM\n(frozen)']
    names = np.hstack([[x]*3 for x in names])

    # plotting the accuracies
    plt.figure(figsize=(4, 5))
    sns.barplot(x=names, y=accuracies, ci='sd', color='gray')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.ylim(0.5, 1)
    plt.tight_layout()
    plt.savefig('./Accuracy_plot_sim2.pdf', dpi=800)
    plt.show()

    # Plotting F1 score from the classification report
    f1_scores = []
    names = []
    for report in r1:
        f1_scores.append(report['3']['f1-score'])
        names.append('1 cell Plexus')
    for report in r4:
        f1_scores.append(report['3']['f1-score'])
        names.append('4 cell Plexus')
    for report in r8:
        f1_scores.append(report['3']['f1-score'])
        names.append('8 cell Plexus')
    for report in r16:
        f1_scores.append(report['3']['f1-score'])
        names.append('16 cell Plexus')
    for report in r_m:
        f1_scores.append(report['3']['f1-score'])
        names.append('Manual Features')
    for report in r_m_nc:
        f1_scores.append(report['3']['f1-score'])
        names.append('Manual Features\n(no network features)')
    for report in rcnn:
        f1_scores.append(report['3']['f1-score'])
        names.append('1D CNN AE')
    for report in rchron:
        f1_scores.append(report['3']['f1-score'])
        names.append('Chronos FM\n(frozen)')

    f1_scores = np.array(f1_scores)
    names = np.array(names)

    # plotting the F1 scores
    plt.figure(figsize=(4, 5))
    sns.barplot(x=names, y=f1_scores, ci='sd', color='gray')
    plt.ylabel('F1 score')
    plt.title('F1 scores for Phenotype 3')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('./f1_score_phenotype_3.pdf', dpi=800)
    plt.show()

    # Plotting F1 score from the classification report
    f1_scores = []
    names = []

    for report in r1:
        f1_scores.append(report['6']['f1-score'])
        names.append('1 cell Plexus')
    for report in r4:
        f1_scores.append(report['6']['f1-score'])
        names.append('4 cell Plexus')
    for report in r8:
        f1_scores.append(report['6']['f1-score'])
        names.append('8 cell Plexus')
    for report in r16:
        f1_scores.append(report['6']['f1-score'])
        names.append('16 cell Plexus')
    for report in r_m:
        f1_scores.append(report['6']['f1-score'])
        names.append('Manual Features')
    for report in r_m_nc:
        f1_scores.append(report['6']['f1-score'])
        names.append('Manual Features \n(no network features)')
    for report in rcnn:
        f1_scores.append(report['6']['f1-score'])
        names.append('1D CNN AE')
    for report in rchron:
        f1_scores.append(report['6']['f1-score'])
        names.append('Chronos FM\n(frozen)')

    f1_scores = np.array(f1_scores)
    names = np.array(names)

    # plotting the F1 scores
    plt.figure(figsize=(4, 5))
    sns.barplot(x=names, y=f1_scores, ci='sd', color='gray')
    plt.ylabel('F1 score')
    plt.title('F1 scores for Phenotype 6')
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('./f1_score_phenotype_6.pdf', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()
