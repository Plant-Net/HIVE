# General imports
import argparse
import os
import random
import matplotlib
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from time import time

# Input scaling imports
from sklearn.preprocessing import MinMaxScaler

# Regression & gene selection imports
import shap
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

# gene condition association imports
import itertools
import matplotlib.pyplot as plt



def filt_low_expr_f(filtered_data, minmax_data):
    """Ulterior filter to exclude all the low expressed genes == read count <= 30

    Parameters
    ----------
    reference_column : first column of the min max scaled input data, containing gene names
    data : raw horizontal integrated dataset
    tr_minmax_data : min max scaled horizontal integrated dataset to be used as input for HIVE VAE

    Returns
    -------
    input_for_regression : input dataframe to be used for the subsequent random forest regression, 
                           filtered out form the very low expression genes
    """

    filt_data = pd.read_csv(filtered_data, sep="\t")
    filt_data.set_index(filt_data.columns[0], inplace=True)

    mm_data = pd.read_csv(minmax_data, sep="\t")
    mm_data.set_index(mm_data.columns[0], inplace=True)
    tr_minmax_data = mm_data.T

    check = {"ref": list(filt_data.index), "min": [], "max": []}

    for r in range(len(filt_data)):
        minim = np.min(list(filt_data.iloc[r]))
        check["min"].append(minim)
        maxim = np.max(list(filt_data.iloc[r]))
        check["max"].append(maxim)

    check_df = pd.DataFrame(check)

    low_exp = check_df[(check_df["min"] <= 30) & (check_df['max'] <= 30)].sort_values(by=["min", "max"])

    # MinMax scaled values for genes with higher expression than those in "low_exp"
    input_for_regression = tr_minmax_data[list(set(tr_minmax_data.columns).difference(set(low_exp.index)))]

    # maintain the order of genes because set is not respecting it
    reorder_columns = [g for g in list(check_df.ref) if g in list(input_for_regression.columns)]
    input_for_regression = input_for_regression[reorder_columns]

    return input_for_regression


def minmax_f(data):
    """MinMax scaling [0,1]
    Parameters
    ----------
    data : horizontal integrated dataset 

    Returns
    -------
    scaled_data_df : min max scaled data from original input data
    """
    scaler = MinMaxScaler()  
    scaled_data = scaler.fit_transform(data)

    scaled_data_df = pd.DataFrame(scaled_data)
    scaled_data_df.set_index(data.index, inplace=True)  # reset index and columns for the scaled data
    scaled_data_df.columns = data.columns

    return scaled_data_df

def rfr_f(smaller_class, reg_X, reg_Y, features, rfr_classes, rfr_output_dir, gini_calcul=False):
    """Random Forest and Importance scores Calculation
    
    Parameters
    ----------
    smaller_class : number of samples in the smaller class for stratified kfold
    reg_X : X input for random forest regression, matrix of minmax values per molecule
    reg_Y : Y input for random forest regression, scaled latent space values 
    features : number of latent features on which to iterate
    rfr_classes : list of assigned labels to samples considered in the same class
    gini_calcul : boolean parameter to change the computation of importance from SHAP to GINI, default=False

    Returns
    -------
    meta : dataframe of rfr performances 
    saves intermediate files of molecule importance per fold and latent feature 
    """
    
    mse = {}
    rmse = {}
    mae = {}
    r2 = {}
    maae = {}


    n_metrics = 5
    # this will results in a final output files containing the performance measures of rfr per fold per latent feature
    meta = {"Fold": list(itertools.chain(*[[f"{i+1}"]*n_metrics for i in range(smaller_class)])),
            "Measure": ["MSE", "RMSE", "MAE", "R2", "MAAE"] * smaller_class}

    skf = StratifiedKFold(n_splits=smaller_class, shuffle=True, random_state=42)
    
    for f in range(len(list(scaled_latent_space.columns))):
        print("Latent Feature", f)
        i = 0
        start = time()
        meta[f"LF{f}"] = []

        best_rmse_index = None
        best_rmse_value = float('inf')

        reg_y = reg_Y[str(f)].values
        for train_index, test_index in skf.split(reg_X, rfr_classes):
            print("fold", i)
            
            x_train = reg_X.iloc[train_index]
            x_test = reg_X.iloc[test_index]
            y_train = reg_y[train_index]
            y_test = reg_y[test_index]
            

            # Random Forest Regression
            rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=11)

            rfr.fit(x_train, y_train)

            ## Use the fitted model to predict the test data
            print("Predictions...")
            y_pred = rfr.predict(x_test)

            # RFR performances
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            maae = metrics.median_absolute_error(y_test, y_pred)

            ########### meta info #########
            meta[f"LF{f}"] += [mse, rmse, mae, r2, maae]

            # SHAP importance calculation just on the best fold

            if rmse < best_rmse_value:
                best_rmse_value = rmse
                best_rmse_index = i

                if gini_calcul:
                    gini_importance = rfr.feature_importances_
                    print("SAVE GINI importance...")
                    gini_df = pd.DataFrame({"ref": input_rfr.columns,
                                            "GINI": gini_importance}).sort_values(by="GINI",
                                                                                  ascending=False,
                                                                                  ignore_index=True)

                    gini_df.to_csv(f"{rfr_output_dir}rfr_GINI_LF{f}.tsv", sep="\t", index=False)

                else:
                    shap_ranked_mean = shap_f(x_test, rfr)
                    # use the below line and follow the plot-comments in shap_f function to plot the shap values
                    # shap_ranked_mean, plot = shap_f(scaled_latent_space, rfr)

                    print("SAVE SHAP scores...")
                    shap_ranked_mean.to_csv(f"{rfr_output_dir}rfr_SHAP_ranked_LF{f}.tsv", sep="\t", index=False)

                

            print("DONE...")
            i += 1

        print("Overall LF time: ", (time() - start) / 60, "minutes")

    return pd.DataFrame(meta)

def shap_f(feat_data, fit_model, show=False):
    """Implementing the SHAP function to be used after random forest regression
    
    Parameters
    ----------
    feat_data : test data in a Cross Validation of RFR model
    fit_model : RFR model from sklearn 
    show : wether to show the SHAP bee plot for each fold of CV or not, default=False

    Returns
    -------
    shap_means_df : dataframe of shap values
    """

    shap.initjs()

    explainer = shap.TreeExplainer(fit_model)
    shap_values = explainer.shap_values(feat_data)

    # a shap value x each gene x each sample x each feature (in for loop the current feature)
    shap_df = pd.DataFrame(shap_values).rename(columns=dict(zip(list(range(0, len(feat_data.columns) + 1)),
                                                                list(feat_data.columns))))

    # mean of the shap values assigned to the same genes in different samples
    shap_means = {}
    for col in list(shap_df.columns):
        shap_means[col] = np.mean(shap_df[col])

    shap_means_df = pd.DataFrame({"ref": list(shap_means.keys()),  # ref
                                  "mean_shap_value": list(shap_means.values())})

    # set show to True and uncomment the following rows and return part to obtain the worm-bees SHAP plot
    # figure = plt.figure()
    # show = True
    # sort values to give a ranked list of shap values from positives to negatives
    return shap_means_df.sort_values(by="mean_shap_value",
                                     ascending=False)  # , shap.summary_plot(shap_values, feat_data, max_display=40, show=show)


def shap_bin_selection_f(n_bins, all_shaps, f_shaps):
    """Binning the shap distribution of each best regression to automatically select important genes

    Parameters
    ----------
    n_bins : number of bist into which divide the SHAP values distribution
    all_shaps : all SHAP values for all Latent Features in the LatentSpace
    f_shaps : SHAP values for current LF

    Returns
    -------
    mean_ref_bin_neg : mean of negative SHAP values of the bin from which to start selecting the genes 
    mean_ref_bin_pos : mean of positive SHAP values of the bin from which to start selecting the genes
    """
    min_shaps = min(all_shaps)  # min of all the shap values x each gene x each feature in the latent space
    max_shaps = max(all_shaps)  # max
    step = (max_shaps - min_shaps) / n_bins  # width of the bin

    bins = list(np.arange(min_shaps, max_shaps, step))  # list of all minimum thresholds of each bin

    genes_in_bins = []  # how many genes falls in each bin

    for i in range(len(bins) - 1):
        genes_in_bins.append(len(list(f_shaps["mean_shap_value"][(f_shaps.mean_shap_value >= bins[i]) & (
                    f_shaps.mean_shap_value < bins[i + 1])])))

    i_biggest = genes_in_bins.index(
        max(genes_in_bins))  # the biggest bin in terms of gene counts fairly contains genes of no interest

    # hive select only genes with shap values in bins smaller or greater than this bins
    ref_bin_neg = f_shaps["mean_shap_value"][
        (f_shaps.mean_shap_value > bins[i_biggest - 1]) & (f_shaps.mean_shap_value <= bins[i_biggest])]
    ref_bin_pos = f_shaps["mean_shap_value"][
        (f_shaps.mean_shap_value > bins[i_biggest + 1]) & (f_shaps.mean_shap_value <= bins[i_biggest + 2])]

    mean_ref_bin_neg = ref_bin_neg.mean()
    mean_ref_bin_pos = ref_bin_pos.mean()

    return mean_ref_bin_neg, mean_ref_bin_pos


def distance_calcul(row):
    """Calculates extreme value of expression and distance between conditions to associate the 
    most estreme condition evidence to the HIVE selected element

    Parameters
    ----------
    row : each row of a dataframe containing absolute log2FoldChange values

    Results
    -------
    distances : dataframe with extra column in which the most extreme distance in expression is found
    """

    max_value = row.max()  # find the extreme value 
    distances = row.apply(lambda x: max_value - x)  # calculate the distance with the extreme value
    
    return distances



if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="%(prog)s [options]",
                                     description="",
                                     epilog="")

    parser.add_argument("-o", "--output_dir",
                        help="output directory in which to save HIVE output files",
                        default="./bHIVE_results/")

    parser.add_argument("-f", "--filtered_data",
                        help="path to the file of filtered read counts",
                        default="./yHIVE_results/yHIVE_filtered_input_data.tsv")

    parser.add_argument("-mm", "--minmax_data",
                        help="path to the file of minmax scaled read counts",
                        default="./yHIVE_results/yHIVE_minmax_scaled_input_data.tsv")

    parser.add_argument("-ls", "--latent_space",
                        help="path to the file regarding the latent space previously obtained",
                        default="./yHIVE_results/yHIVE_latent_space.tsv")

    parser.add_argument("-c", "--rfr_classes",
                        help="indicate the vector corresponding to classes in your experimental design for the random forest regression step,"
                             " MAINTAIN THE SAMPLE ORDER, e.g. 0 0 1 1 1 2 2",
                        nargs="+",
                        type=int,
                        default=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 0, 2])

    parser.add_argument("-ga", "--gene_association",
                        help="indicate whether or not to perform also the selection (e.g. gene) association to condition",
                        action="store_true")

    parser.add_argument("-fc", "--fold_change",
                        help="path to log2FoldChange file of HIVE selected genes")
    
    parser.add_argument("-sep", "--separator",
                        help="indicate the separator character for columns in input data",
                        type=str,
                        default=",")

    parser.add_argument("-M", "--merge_cols",
                        help="indicate wether there is the need to merge columns of l2fc matrix for final molecule association to condition, e.g. replicates",
                        action="store_true")

    parser.add_argument("-cM", "--cols_to_merge",
                        help="list of lists of names of columns to be merged, e.g. 'as3' 'as6' 'as9'",
                        type=str,
                        nargs="+",
                        action="append") #,
                        # default=[['as3', 'as6', 'as9'], ['ad3', 'ad6', 'ad9']])
    
    parser.add_argument("-fM", "--final_cols",
                        help="list of names of final merged columns, e.g. 'as' 'ad'",
                        type=str,
                        nargs="+",
                        action="append") #,
                        # default=['as', 'ad'])


    args = parser.parse_args()
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    smaller_class = np.min(list(Counter(args.rfr_classes).values()))
    print(f"START RFR-{smaller_class}foldCV")

    # # ########################################################
    # # #### RANDOM FOREST REGRESSION WITH CROSS VALIDATION ####
    # # ########################################################
    # # # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

    input_rfr = filt_low_expr_f(args.filtered_data, args.minmax_data)
    # save Random Forest Regression input
    input_rfr.to_csv(f"{args.output_dir}gHIVE_NoLowExp_rfr_input_data.tsv", sep="\t")

    # parse latent space
    latent_space = pd.read_csv(args.latent_space, sep="\t")
    latent_space_feat = latent_space.set_index(latent_space.columns[0])
    # MinMax scale latent features to be in the same value-magnitude of the genes that will be regressed
    scaled_latent_space = minmax_f(latent_space_feat)

    reg_X = input_rfr.copy()
        

    rfr_output_dir = f"{args.output_dir}CV_{smaller_class}fold/SHAP_per_fold/"
    if not os.path.exists(rfr_output_dir):
        os.makedirs(rfr_output_dir)

    meta_df = rfr_f(smaller_class, reg_X, scaled_latent_space, list(scaled_latent_space.columns), args.rfr_classes, rfr_output_dir)
    
    # SHAP SELECTION
    print("START HIVE-SHAP SELECTION")
    all_bestf_shaps = []
    for f in range(len(scaled_latent_space.columns)):
        file_shaps = pd.read_csv(f"{rfr_output_dir}rfr_SHAP_ranked_LF{f}.tsv", sep="\t")
        shap_values = file_shaps.mean_shap_value
        all_bestf_shaps += list(shap_values)

    selected_shap_neg = {}  # per feature gene shap values top negatives
    selected_shap_pos = {}  # and positives
    store_ranking = {}

    # create a dictionary of dictionaries where the first keys are the positions of genes and the values will become
    # the gene in that position for each feature associated to the corresponding shap value
    for p in range(len(input_rfr.columns)):
        store_ranking[f"pos{p + 1}"] = {}

    for f in range(len(scaled_latent_space.columns)):
        f_shaps = pd.read_csv(f"{rfr_output_dir}rfr_SHAP_ranked_LF{f}.tsv",
                                sep="\t")
        neg, pos = shap_bin_selection_f(60, all_bestf_shaps, f_shaps)  # change the number of bins

        selected_shap_neg[f] = list(
            f_shaps["ref"][f_shaps.mean_shap_value <= neg])  # all genes in the smaller bins below the neg shap mean
        selected_shap_pos[f] = list(
            f_shaps["ref"][f_shaps.mean_shap_value >= pos])  # all genes in the higher bins above the pos shap mean
        
        # GENE RANKING PER FEATURE
        sel_f = selected_shap_pos[f] + selected_shap_neg[f]
        shap_f_sel = f_shaps[f_shaps.ref.isin(sel_f)].copy()
        shap_f_sel["mean_shap_value"] = [abs(el) for el in list(shap_f_sel["mean_shap_value"])]
        shap_f_rank = shap_f_sel.sort_values(by="mean_shap_value", ascending=False, ignore_index=True)
        
        for p in range(len(shap_f_rank)):  # this is just a control for the real number of positions in the dataset
            # can happen that a gene has the same position in two or more features so always keep the higher value
            # for future ranking
            
            if shap_f_rank["ref"].iloc[p] not in list(store_ranking[f"pos{p + 1}"].keys()):
                store_ranking[f"pos{p + 1}"][shap_f_rank["ref"].iloc[p]] = shap_f_rank["mean_shap_value"].iloc[p]

            else:
                if store_ranking[f"pos{p + 1}"][shap_f_rank["ref"].iloc[p]] < shap_f_rank["mean_shap_value"].iloc[p]:
                    store_ranking[f"pos{p + 1}"][shap_f_rank["ref"].iloc[p]] = shap_f_rank["mean_shap_value"].iloc[p]
                else:
                    pass

    # FINAL GENE SELECTION
    # for each position sort the absolute value of each gene and append to a single final list
    ranked_selected_genes = []
    for p in list(store_ranking.keys()):
        tmp_df = pd.DataFrame({"ref": store_ranking[p].keys(),
                                "shap_abs_value": store_ranking[p].values()})
        tmp_df_rank = tmp_df.sort_values(by="shap_abs_value", ascending=False, ignore_index=True)
        for g in list(tmp_df_rank.ref):
            # if a gene is already present in the list means that in at least one features it already have a major
            # rank order so is not appended again
            if g not in ranked_selected_genes:
                ranked_selected_genes.append(g)
            else:
                pass

    # HIVE OUTPUT RANKED SELECTED GENES
    with open(f"{args.output_dir}gHIVE_selection_shap.txt", "w") as hive_out:
        for g in ranked_selected_genes:
            hive_out.write(f"{g}\n")
    print("END SELECTION")
   

    ##################################
    #### ASSOCIATION TO CONDITION ####
    ##################################
    if args.gene_association:
        print("START ASSOCIATION TO CONDITION")
        l2fc = pd.read_csv(args.fold_change, sep=args.separator, index_col=0).fillna(0)
        abs_l2fc = abs(l2fc)
        
        # find extreme association
        distances = abs_l2fc.apply(distance_calcul, axis=1)
        distances['extreme_cond'] = abs_l2fc.idxmax(axis=1)

        # define thresholds 
        all_values = []

        for index, row in distances.iterrows():
            extreme_cond = row['extreme_cond']
            values_except_max_and_cond = row.drop([extreme_cond, 'extreme_cond'])
            all_values.extend(values_except_max_and_cond)

        min_val = min(all_values)
        max_val = max(all_values)
        mean_val = np.mean(all_values)
        sigmas_ori = np.std(all_values)
        # sigmas = 1
        sigmas = sigmas_ori/2

        ### plot
        plt.figure(figsize=(10, 8))  
        plt.hist(all_values, bins=100) 
        plt.axvline(sigmas, color='red', linestyle='--', label='Half-Sigma')
        plt.axvline(sigmas_ori, color='green', linestyle='--', label='Sigma')
        plt.legend(loc="upper right")
        plt.title('Distances distribution')
        plt.grid(True) 
        plt.savefig(f"{args.output_dir}gHIVE_distance_distribution.pdf")
        ###

        # association extremes to ones passing the threshold
        distance_columns = distances.copy()
        
        for index, row in distances.iterrows():
            cond_extreme = row['extreme_cond']
            columns_to_exclude = [cond_extreme, 'extreme_cond']
            distance_columns.loc[index, columns_to_exclude] = np.nan

        outside_sigmas = distance_columns.loc[distance_columns.fillna(float('inf')).iloc[:, :-1].gt(sigmas).all(axis=1)]
        outside_sigmas_df = pd.DataFrame(index=outside_sigmas.index, columns=outside_sigmas.columns).fillna(0)

        l2fc_cond_spe = l2fc.loc[outside_sigmas_df.index]

        max_abs_column_index = l2fc_cond_spe.abs().idxmax(axis=1)
        cond_spe = pd.DataFrame(0, index=l2fc_cond_spe.index, columns=l2fc_cond_spe.columns)

        for i, col_name in enumerate(max_abs_column_index):
            cond_spe.loc[cond_spe.index[i], col_name] = 1

        distance_columns['extreme_cond'] = distances['extreme_cond']

        # pariwise association
        cols = l2fc.columns.tolist()
        pairs = list(itertools.combinations(cols, 2))
        couples = [f'{pair[0]}-{pair[1]}' for pair in pairs]

        results = pd.DataFrame(index=distance_columns.index, columns=couples)
        
        for idx, row in distance_columns.iterrows():
            max_cond = row.iloc[-1]
            
            couples_max = [couple for couple in couples if max_cond in couple]
            
            for el in couples_max:
                cond1, cond2 = el.split('-')
                if cond1 != max_cond :
                    results.loc[idx, el] = row[cond1] 
                else :
                    results.loc[idx, el] = row[cond2]
                    
        results['extreme_cond'] = distances['extreme_cond']
        
        # filtering
        filtered_values = {}
        df_list = []

        i = 1
        for couple in couples:
            
            couple_df = results[[couple]]
            filtered_values[i] = couple_df[couple_df.apply(lambda row: row <= sigmas).all(axis=1)]
            df_list.append(filtered_values[i])
            i += 1 

        df_all = pd.concat(df_list, axis=1)
        
        ## Binarization
        binarized = pd.DataFrame(index=df_all.index, columns=df_all.columns)

        for i in range(len(df_all.index)):
            for col in df_all.columns:
                if pd.isna(df_all[col][i]):
                    binarized[col][i] = 0
                else:
                    binarized[col][i] = 1
        
        df_conds = pd.DataFrame(columns=l2fc.columns) 

        for index, row in binarized.iterrows():
            conds_presence = [0]*len(df_conds.columns)
            
            for col in binarized.columns:
                conds = col.split('-')
                if row[col] == 1:
                    for cond in conds:
                        if cond in df_conds.columns:
                            conds_presence[df_conds.columns.get_loc(cond)] = 1
                
            df_conds.loc[index] = conds_presence
        
        df_conds_all = pd.concat([df_conds, cond_spe])
        df_conds_all.to_csv(f'{args.output_dir}gHIVE_binarized_selection_associations.tsv', sep="\t")

        # merge conditions
        if args.merge_cols:
            for col_set, el in enumerate(args.final_cols):
                df_conds_all[el] = 0

                mask_as = df_conds_all[args.cols_to_merge[col_set]].any(axis=1)
                

                df_conds_all.loc[mask_as, el] = 1
                
                
                df_conds_all = df_conds_all.drop(columns=list(itertools.chain(*args.cols_to_merge)))
                df_conds_all.to_csv(f'{args.output_dir}gHIVE_merged_binarized_selection_associations.tsv', sep="\t")
        
        # association to all combination of conditions
        df_conds_all['Associated_conds'] = df_conds_all.apply(lambda row: df_conds_all.columns[row == 1].tolist(), axis=1)
        df_conds_list = df_conds_all[['Associated_conds']]
        df_conds_list.to_csv(f'{args.output_dir}gHIVE_selection_associations.tsv', sep="\t")
        
        # count different possible associations
        counts = df_conds_list['Associated_conds'].value_counts()
        counts.to_csv(f'{args.output_dir}gHIVE_counts_selection_associations.tsv', sep="\t")
    
    else:
        pass