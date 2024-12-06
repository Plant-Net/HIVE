# General imports
import argparse
import os
import random
import matplotlib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# phenotypic correlation imports
import math
import scikit_posthocs
import scipy
from scipy.stats import kruskal
from sklearn import preprocessing



if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="%(prog)s [options]",
                                     description="",
                                     epilog="")

    parser.add_argument("-o", "--output_dir",
                        help="output directory in which to save HIVE output files, e.g. /home/HIVE_results/",
                        default="./bHIVE_results")

    parser.add_argument("-ls", "--latent_space",
                        help="path to the latent space file previously obtained with the yHIVE1.0.py script",
                        default="./yHIVE_results/yHIVE_latent_space.tsv")

    parser.add_argument("--pheno",
                    help="indicate a numeric vector with numbers assigned to each sample (DO NOT CONSIDER CONTROLS)"
                            "corresponding to phenotypic characteristics of interest, following their original order, "
                            "to explore the pheno. char. captured by latent features",
                    type=str,
                    nargs="+")


    args = parser.parse_args()
    
    
    args.output_dir = args.output_dir 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    latent_space = pd.read_csv(args.latent_space, sep="\t", index_col=0)
    latent_space["pheno"] = args.pheno
    
    ls_pheno = latent_space[~latent_space.pheno.str.contains("ctrl") == True]
    label_enc = preprocessing.LabelEncoder()
    ls_pheno["pheno"] = label_enc.fit_transform(ls_pheno["pheno"])

    if len(np.unique(ls_pheno.pheno)) <= 2:
        # point biserial correlation
        i = 0
        stress_f_corr = {"LF": [], "p-value": [], "correlation": []}
        for f in list(latent_space.columns)[:-1]:  # last columns is pheno, here we are interested into iterate over latent features 
            if scipy.stats.pointbiserialr(list(ls_pheno["pheno"]), list(ls_pheno[f]))[1] <= 0.01:
                i += 1
                stress_f_corr["LF"].append(f"LF{f}")
                stress_f_corr["p-value"].append(scipy.stats.pointbiserialr(ls_pheno.pheno, ls_pheno[f])[1])
                stress_f_corr["correlation"].append(scipy.stats.pointbiserialr(ls_pheno.pheno, ls_pheno[f])[0])

        stress_f_corr_df = pd.DataFrame(stress_f_corr).sort_values(by="p-value")
        stress_f_corr_df.to_csv(f"{args.output_dir}bHIVE_LFs_pbisercorr_sig_p001.tsv", sep="\t", index=False)

    else:
        
        # Kruskal-Wallis test
        kw_dict = {}
        with open(f"{args.output_dir}bHIVE_LFs_kruskalwallis.txt", "w") as kw_out:
            for f in list(latent_space.columns)[:-1]:
                groups = [ls_pheno[f][ls_pheno.pheno == g] for g in np.unique(ls_pheno.pheno)]
                kw = kruskal(*groups)
                kw_out.write(f"Kruskal-Wallis LF{f}: {kw}\n")
                kw_dict[f"LF{f}"] = kw[1] # for later usage

        # Dunns test
        dnn_dir = args.output_dir + "bHIVE_Dunns_test/"
        if not os.path.exists(dnn_dir):
            os.makedirs(dnn_dir)

        dnn_dict = {}
        for f in list(latent_space.columns)[:-1]:
            dnn = scikit_posthocs.posthoc_dunn(ls_pheno, val_col=f, group_col="pheno")
            with open(f"{dnn_dir}LF{f}_dunn_metrix.txt",  "w") as dunn_out:
                dunn_out.write(dnn.to_string())
            dnn_dict[f"LF{f}"] = {}
            for i in range(len(dnn.index)):
                for j in range(i + 1, len(dnn.columns)):
                    couple = f"{dnn.index[i]}/{dnn.columns[j]}"
                    v = dnn.iloc[i, j]
                    dnn_dict[f"LF{f}"][couple] = v

            # -log(10) p-value transformation
            f_dnn_log = pd.DataFrame({f"LF{f}": list(dnn_dict[f"LF{f}"].keys()),
                                        "p-value": list(dnn_dict[f"LF{f}"].values())})

            f_dnn_log["-log10(p-value)"] = -np.log10(f_dnn_log["p-value"])

            f_dnn_log.to_csv(f"{dnn_dir}LF{f}_dunn_log_pval.tsv", sep="\t", index=True)

        # all Dunns' test in one dataframe
        dnn_df = pd.DataFrame(dnn_dict)
        dnn_df.to_csv(f"{args.output_dir}LFs_dunn.tsv", sep="\t")
        
        # combine results of both tests sorting LFs by KW p-values
        kw_sort = pd.DataFrame({"LFs": list(kw_dict.keys()),
                                "p-value": list(kw_dict.values())}).sort_values(by="p-value")
        
        dnn_df[list(kw_sort.LFs)].to_csv(f"{args.output_dir}bHIVE_LFs_kw_dunn_corr.tsv", sep='\t')