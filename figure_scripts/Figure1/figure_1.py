import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns


def plot_and_save_timeseries(gcamp_signal: np.ndarray,
                             plot_color: str,
                             fs: int,
                             filename: str):
    gcamp_signal = gcamp_signal[:30]
    print(gcamp_signal.shape)
    print(fs)
    time = np.arange(gcamp_signal.shape[1]) / fs
    print(time)
    plt.figure(figsize=(6, 6))
    sig_max = np.amax(gcamp_signal) * 0.25
    print(np.amax(gcamp_signal))
    for c, cell in enumerate(gcamp_signal):
        cell_scaled = (cell / sig_max) + c
        plt.plot(time, cell_scaled, color='k')
        plt.fill_between(time, c, cell_scaled, color=plot_color, alpha=.4)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Number')
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', dpi=800)
    plt.show()


def main():
    ttx_dmso_df = pd.read_csv('ttx_vs_dmso_time_series.csv')
    time = ttx_dmso_df['Time']
    dmso = ttx_dmso_df['DMSO']
    ttx = ttx_dmso_df['TTX']

    trace_color = "#738678"
    plt.rcParams['text.usetex'] = False
    # Setting the font to arial
    plt.rcParams['font.family'] = 'Sans Serif'
    # Setting the font size
    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(6, 2))
    plt.plot(time, dmso, color=trace_color)
    plt.ylim(0, 0.15)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\frac{\Delta F}{F}$')
    plt.tight_layout()
    plt.savefig('figure_1_dmso_plot.pdf', dpi=800)
    plt.show()

    plt.figure(figsize=(6, 2))
    plt.plot(time, ttx, color=trace_color)
    plt.ylim(0, 0.15)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\frac{\Delta F}{F}$')
    plt.tight_layout()
    plt.savefig('figure_1_ttx_plot.png', dpi=800)
    plt.show()

    dmso_traces = np.load("dmso_traces.npy")
    tent_traces = np.load("tent_traces.npy")
    sampling_freq = 25
    plot_and_save_timeseries(dmso_traces, trace_color, sampling_freq, 'figure_1f_dmso')
    plot_and_save_timeseries(tent_traces, trace_color, sampling_freq, 'figure_1f_tent')


    adata_dmso = ad.read_h5ad('./../../plexus_data_archive/figure1_data/DMSO_traces.h5ad')
    adata_tent = ad.read_h5ad('./../../plexus_data_archive/figure1_data/DMSO_traces.h5ad')

    correlation_coeffs = []
    treatments = []
    for rep in adata_dmso.obs['replicate'].unique():
        dmso_rep = adata_dmso[adata_dmso.obs['replicate'] == rep]
        corr = np.mean(np.corrcoef(dmso_rep.X, dmso_rep.X))
        correlation_coeffs.append(corr)
        treatments.append('DMSO')

    for rep in adata_tent.obs['replicate'].unique():
        tent_rep = adata_tent[adata_tent.obs['replicate'] == rep]
        corr = np.mean(np.corrcoef(tent_rep.X, tent_rep.X))
        correlation_coeffs.append(corr)
        treatments.append('TeNT')

    corr_df = pd.DataFrame({'Correlation Coefficient': correlation_coeffs, 'Treatment': treatments})
    plt.figure(figsize=(7, 5))
    sns.violinplot(x='Treatment', y='Correlation Coefficient', data=corr_df, color=trace_color)
    sns.swarmplot(x='Treatment', y='Correlation Coefficient', data=corr_df, color='k', size=3)
    plt.ylabel('Field of view Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('figure_1g_correlation_plot.pdf', format='pdf', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()
