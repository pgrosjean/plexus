import anndata as ad
import seaborn as sns
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_roc_curves_with_ci(X_dict, y_dict, groups_dict, color_dict, n_splits=2):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for treatment, X in X_dict.items():
        y = y_dict[treatment]
        groups = groups_dict[treatment]
        color = color_dict[treatment]
        
        cv = GroupKFold(n_splits=n_splits)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv.split(X, y, groups)):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X[train], y[train])

            y_pred = model.predict_proba(X[test])[:, 1]
            fpr, tpr, _ = roc_curve(y[test], y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = roc_auc_score(y[test], y_pred)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color=color,
                label=f'{treatment} vs. Neg Ctrl (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    return fig


def main():
    palette = sns.color_palette(['#8ea1ab', 
                             '#d6278a', 
                             '#14db5d', 
                             '#d67c0d',
                             '#7c4fd6'])
    hue_order = ['Negative Control', '2 mM Ca2+', '2 mM Mg2+','TeNT', 'TTX']
    adata_stimulation = ad.read_h5ad("../../plexus_data_archive/plexus_embeddings/neuroactive_stimulation/neuroactive_stimulation_manual_features.h5ad")
    adata_stimulation_wt = adata_stimulation[adata_stimulation.obs["cell_line"] == "WTC11-WT-Tau"]
    adata_stimulation_wt.obs['location_id'] = adata_stimulation_wt.obs['for_aggregation'].apply(lambda x: '-'.join(x.split('-')[:-2]))
    standard_scaler = StandardScaler()
    X = adata_stimulation_wt.X
    X = np.nan_to_num(X, nan=-1)
    adata_x_scaled_wt = standard_scaler.fit_transform(X)
    adata_stimulation_wt.X = adata_x_scaled_wt
    X_dict = {}
    y_dict = {}
    groups_dict = {}
    treatment_nt = 'Negative Control'
    treatments_to_test = [t for t in adata_stimulation_wt.obs['treatment'].unique() if t != treatment_nt]
    for treatment in treatments_to_test:
        adata_d14_wt_treat_temp = adata_stimulation_wt[adata_stimulation_wt.obs['treatment'].isin([treatment, treatment_nt])]
        X_dict[treatment] = adata_d14_wt_treat_temp.X
        y_dict[treatment] = adata_d14_wt_treat_temp.obs['treatment'].apply(lambda x: 1 if x == treatment else 0).values
        groups_dict[treatment] = list(adata_d14_wt_treat_temp.obs['location_id'].values)
    plt.rcParams.update({'font.size': 16})
    treat_color_dict = {k: v for k, v in zip(hue_order, palette)}
    _ = plot_roc_curves_with_ci(X_dict, y_dict, groups_dict, treat_color_dict, 3)
    plt.savefig('roc_curves_neuroactive_stimulation_manual_features.pdf', dpi=800)
    plt.show()

    adata_stim_embed = ad.read_h5ad("../../plexus_data_archive/plexus_embeddings/neuroactive_stimulation/neuroactive_stimulation_plexus_embeddings.h5ad")
    adata_stim_embed_wt = adata_stim_embed[adata_stim_embed.obs["cell_line"] == "WTC11-WT-Tau"]
    adata_stim_embed_wt.obs['location_id'] = adata_stim_embed_wt.obs['condition'].astype('str') + '-' + adata_stim_embed_wt.obs['well_id'].astype('str')

    X_dict = {}
    y_dict = {}
    groups_dict = {}
    treatment_nt = 'Negative Control'
    treatments_to_test = [t for t in adata_stim_embed_wt.obs['treatment'].unique() if t != treatment_nt]

    for treatment in treatments_to_test:
        adata_d14_wt_treat_temp = adata_stim_embed_wt[adata_stim_embed_wt.obs['treatment'].isin([treatment, treatment_nt])]
        X_dict[treatment] = adata_d14_wt_treat_temp.X
        y_dict[treatment] = adata_d14_wt_treat_temp.obs['treatment'].apply(lambda x: 1 if x == treatment else 0).values
        groups_dict[treatment] = list(adata_d14_wt_treat_temp.obs['location_id'].values)

    _ = plot_roc_curves_with_ci(X_dict, y_dict, groups_dict, treat_color_dict)
    plt.savefig('roc_curves_neuroactive_stimulation_plexus_embeddings.pdf', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()
