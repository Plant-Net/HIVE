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

# VAE imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
import scipy
from scipy.stats import entropy

# Regression & gene selection imports
import shap
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# phenotypic correlation imports
import math
import scikit_posthocs
from scipy.stats import kruskal

# gene condition association imports
import itertools

tf.random.set_seed(779)


def horizontal_integration(input_dir, reference_column, suff, sep, header=0):
    """
    Merge each file with the next one maintaining the raw order (typically gene order) of the first file.
    The first column of all file must be called "ref" and has to contain the gene names.
    """

    file_list = []
    for el in os.listdir(input_dir):
        if el.endswith(suff):
            file_list.append(el)

    for i, f in enumerate(file_list):
        if i < len(file_list) - 1:
            act_df = pd.read_csv(os.path.join(input_dir, f), sep=sep, header=header)
            next_df = pd.read_csv(os.path.join(input_dir, file_list[i + 1]), sep=sep, header=header)
            merge_tmp_df = act_df.merge(next_df, on=reference_column)

        elif i == len(file_list) - 1:
            act_df = pd.read_csv(os.path.join(input_dir, f), sep=sep, header=header)
            merge_tmp_df = merge_tmp_df.merge(act_df, on=reference_column)

            return merge_tmp_df

        else:
            pass


def prefiltering_f(data):
    """
    Filter out all rows with variance below first distribution quartile
    """

    data_no_zeroes = data

    variances = {}
    for r in range(len(data_no_zeroes)):
        arr = np.array(data_no_zeroes.iloc[r])[1:]
        var = np.var(arr)
        variances[data_no_zeroes.ref.iloc[r]] = var
    variances_df = pd.DataFrame({"ref": list(variances.keys()),
                                 "variance": list(variances.values())})
    thr_var = float(variances_df["variance"].quantile(0.25, interpolation='lower'))

    var_to_retain = variances_df[variances_df["variance"] > thr_var]

    to_retain = data_no_zeroes[data_no_zeroes.ref.isin(list(var_to_retain.ref))]

    return to_retain


def minmax_hive_f(data):
    """
    MinMax scaling [0,1]
    """
    scaler = MinMaxScaler()  #
    scaled_data = scaler.fit_transform(data)

    scaled_data_df = pd.DataFrame(scaled_data)
    scaled_data_df.set_index(data.index, inplace=True)  # reset index and columns for the scaled data
    scaled_data_df.columns = data.columns

    return scaled_data_df


def sampling_f(inputs):
    """
    sample the latent space from the latent distribution
    """
    ld_mean, ld_log_var = inputs
    batch = keras.backend.shape(ld_mean)[0]
    dim = keras.backend.int_shape(ld_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim), seed=779)

    z = ld_mean + keras.backend.exp(0.5 * ld_log_var) * epsilon
#     print("sampling out:", z)
    return z


def setup_encoder_f(data, el1=5000, el2=1000, el3=500, kernel="lecun_normal", ls_size=80):
    """
    the encoder is implemented to have 3 layers = [el1, el2, el3] and a latent space size of 80 latent features
    the latent space is sampled from the latent disribution (because of the autoencoder type)
    """

    encoder_input = keras.Input(shape=(data.shape[1],), name="encoder_input")

    encoder_l1 = layers.Dense(el1, kernel_initializer=kernel, name="encoder_layer1")(encoder_input)
    el1_normalization = layers.BatchNormalization(name="encoder_layer1_norm")(encoder_l1)
    encoder_l1_activation = layers.ELU()(el1_normalization)

    encoder_l2 = layers.Dense(el2, kernel_initializer=kernel, name="encoder_layer2")(encoder_l1_activation)
    el2_normalization = layers.BatchNormalization(name="encoder_layer2_norm")(encoder_l2)
    encoder_l2_activation = layers.ELU()(el2_normalization)

    encoder_l3 = layers.Dense(el3, kernel_initializer=kernel, name="encoder_layer3")(encoder_l2_activation)
    el3_normalization = layers.BatchNormalization(name="encoder_layer3_norm")(encoder_l3)
    encoder_l3_activation = layers.ELU()(el3_normalization)

    return encoder_input, encoder_l3_activation


def latent_space_f(encoder_input, encoder_l3_activation, sampling, ls_size=80):
    """
    uses the output of the encoder setup function to create the encoder model
    """

    ld_mean = layers.Dense(ls_size, name="ld_mean")(encoder_l3_activation)
    ld_log_var = layers.Dense(ls_size, name="ld_log_var")(encoder_l3_activation)
    sampling_inputs = [ld_mean, ld_log_var]

    ls = layers.Lambda(sampling, output_shape=(ls_size,), name="ls")(sampling_inputs)  # latent space vector

    return sampling_inputs, ls


def encoder_model_f(encoder_input, sampling_inputs, ls):
    """
    build-up the encorder model
    """
    encoder_model = keras.Model(inputs=encoder_input, outputs=sampling_inputs + [ls], name="encoder")

    # uncomment the last return to have a summary of the model
    return encoder_model, encoder_model.summary()


def setup_decoder_f(ls_size=80, dl1=500, dl2=1000, dl3=5000, kernel="lecun_normal"):
    """
    the decoder has the same and inverse structure of the encoder, thus, 3 layers = [dl1, dl2, dl3]
    """
    decoder_input = keras.Input(shape=(ls_size,), name="decoder input")

    decoder_l1 = layers.Dense(dl1, kernel_initializer=kernel, name="decoder_layer1")(decoder_input)
    dl1_normalization = layers.BatchNormalization(name="decoder_layer1_norm")(decoder_l1)
    decoder_l1_activation = layers.ELU()(dl1_normalization)

    decoder_l2 = layers.Dense(dl2, kernel_initializer=kernel, name="decoder_layer2")(decoder_l1_activation)
    dl2_normalization = layers.BatchNormalization(name="decoder_layer2_norm")(decoder_l2)
    decoder_l2_activation = layers.ELU()(dl2_normalization)

    decoder_l3 = layers.Dense(dl3, kernel_initializer=kernel, name="decoder_layer3")(decoder_l2_activation)
    dl3_normalization = layers.BatchNormalization(name="decoder_layer3_norm")(decoder_l3)
    decoder_l3_activation = layers.ELU()(dl3_normalization)

    return decoder_input, decoder_l3_activation


def decoder_model_f(data, decoder_input, decoder_l3_activation):
    """
    build-up the decorder model
    """
    reconstruction = layers.Dense(data.shape[1])(decoder_l3_activation)  # reconstruction layer (final obj of decoder)
    decoder_model = keras.Model(decoder_input, reconstruction, name="decoder")
    # uncomment the last return to have a summary of the model
    return reconstruction, decoder_model, decoder_model.summary()


def msle_loss_f(data, encoder_input, outputs):
    msle = keras.losses.MSLE(encoder_input, outputs[0])
    msle *= data.shape[1]
    return msle


def kl_loss_f(ld_mean, ld_log_var, kl_loss_weight=0.5):
    kl_loss = 1 + ld_log_var - keras.backend.square(ld_mean) - keras.backend.exp(ld_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -kl_loss_weight
    return kl_loss


def vae_loss_f(msle, kl_loss):
    vae_loss = keras.backend.mean(msle + kl_loss)
    return vae_loss


def vae_init_f(encoder_input, outputs, ls, vae_loss):
    """
    initialization of the VAE model using encoder and decoder defined before
    """
    vae_model = keras.Model(encoder_input, [outputs, ls], name="vae")
    vae_model.add_loss(vae_loss)
    vae_model.compile(keras.optimizers.Adam())
    # uncomment the last return to have a summary of the model
    return vae_model  # , vae_model.summary()


def vae_f(vae, data, epochs=300, batch_size=32):
    """
    actually fitting the VAE on the data
    """
    labels = np.array(data.index)
    vae_fitting = vae.fit(data, labels, epochs=epochs, batch_size=batch_size)
    predictions, latent_space = vae.predict(data)
    latent_space = pd.DataFrame(latent_space).set_index(labels)  # the output needed for further analyses
    return vae_fitting, predictions, latent_space


def perplexity_f(y_true, y_pred):
    true_len = y_true.shape[0]
    pred_len = y_pred.shape[0]
    pad_length = pred_len - true_len
    if pad_length > 0:
        y_true = pad_sequences([y_true], maxlen=pred_len, padding='post', truncating='post')[0] # keras
    elif pad_length < 0:
        y_pred = pad_sequences([y_pred], maxlen=true_len, padding='post', truncating='post')[0]
    cross_entropy = entropy(y_true, y_pred)
    perpl = np.power(2.0, cross_entropy)
    return np.mean(perpl, axis=0)


def mse_f(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    return np.mean(mse, axis=0)


def mae_f(y_true, y_pred):
    mae = keras.losses.MeanAbsoluteError()
    mae_score = np.square(mae(y_true, y_pred))
    return np.mean(mae_score, axis=0)


def kl_divergence_f(y_true, y_pred):

    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    kl_div = tf.reduce_sum(y_true_tensor * tf.math.log(y_true_tensor / y_pred_tensor))
    return np.mean(kl_div, axis=0)


def rmsle_f(y_true, y_pred):
    y_true_log = tf.math.log1p(y_true)
    y_pred_log = tf.math.log1p(y_pred)
    rmsle = tf.sqrt(tf.reduce_mean(tf.square(y_true_log - y_pred_log)))
    return np.mean(rmsle, axis=0)


def loss_plot_f(fitting):
    plt.plot(fitting.epoch, fitting.history["loss"], 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    return plt.show()


def filt_low_expr_f(reference_column, data):
    """
    Ulterior filter to exclude all the low expressed genes == read count <= 30
    """

    check = {f"{reference_column}": list(data[f"{reference_column}"]), "min": [], "max": []}

    for r in range(len(data)):
        minim = np.min(list(data.iloc[r])[1:])
        check["min"].append(minim)
        maxim = np.max(list(data.iloc[r])[1:])
        check["max"].append(maxim)

    check_df = pd.DataFrame(check)

    low_exp = check_df[(check_df["min"] <= 30) & (check_df['max'] <= 30)].sort_values(by=["min", "max"])

    # MinMax scaled values for genes with higher expression than those in "low_exp"
    input_for_regression = tr_data[list(set(tr_data.columns).difference(set(low_exp.ref)))]

    # maintain the order of genes because set is not respecting it
    reorder_columns = [g for g in list(check_df.ref) if g in list(input_for_regression.columns)]
    input_for_regression = input_for_regression[reorder_columns]

    return input_for_regression


def shap_f(feat_data, fit_model, show=False):
    """
    Implementing the SHAP function to be used after regression
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

    shap_means_df = pd.DataFrame({"ref": list(shap_means.keys()),  # gene_name
                                  "mean_shap_value": list(shap_means.values())})

    # set show to True and uncomment the following rows and return part to obtain the worm-bees SHAP plot
    # figure = plt.figure()
    # show = True
    # sort values to give a ranked list of shap values from positives to negatives
    return shap_means_df.sort_values(by="mean_shap_value",
                                     ascending=False)  # , shap.summary_plot(shap_values, feat_data, max_display=40, show=show)


def shap_bin_selection_f(n_bins, all_shaps, f_shaps):
    """
    binning the shap distribution of each regression to automatically select important genes
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
    print(genes_in_bins)
    return mean_ref_bin_neg, mean_ref_bin_pos


def calc_distance_f(row):
    max_val = row.max()  # find the maximum l2fc value for the current gene
    diff = row.apply(lambda x: max_val - x)  # calculate the difference btw max_val and all the other l2fc values
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="%(prog)s [options]",
                                     description="",
                                     epilog="")
    parser.add_argument("-i", "--input_dir",
                        help="input directory in which to find the raw expression data for HIVE application, "
                             "e.g. /home/HIVE_input_data/",
                        default="./")
    parser.add_argument("-o", "--output_dir",
                        help="output directory in which to save HIVE output files, e.g. /home/HIVE_results/",
                        default="./HIVER_results")

    parser.add_argument("-hz", "--horizontal",
                        help="indicate if an horizontal interation of two or more dataset is needed",
                        action="store_true")

    parser.add_argument("-rc", "--reference_column",
                        help="indicate the name of the column containing the gene names",
                        # type=str,
                        default="ref")

    parser.add_argument("-ff", "--file_format",
                        help="indicate the input file/s extension",
                        # type=str,
                        default=".csv")

    parser.add_argument("-sep", "--separator",
                        help="indicate the separator character for columns in input data",
                        # type=str,
                        default=",")

    parser.add_argument("--minmax",
                        help="indicate wether to apply or not the MinMaxScaler [0,1] range",
                        action="store_true",
                        # type=bool,
                        default=True)

    parser.add_argument("-c", "--rfr_classes",
                        help="indicate the vector corresponding to classes in your experimental design, MAINTAIN THE "
                             "SAMPLE ORDER",
                        # type=list,
                        default=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2, 0, 2, 0, 0,
                                 0, 2, 2, 2, 0, 2])

    parser.add_argument("--shap",
                        help="indicate if you want to use SHAP as feature importance calculation for HIVE gene selection",
                        action="store_true")

    parser.add_argument("--gini",
                        help="indicate if you want to use SHAP as feature importance calculation for HIVE gene selection",
                        action="store_true")

    parser.add_argument("--pheno",
                        help="indicate a numeric vector with numbers assigned to each sample (DO NOT CONSIDER CONTROLS)"
                             "corresponding to phenotypic characteristics of interest, following their original order, "
                             "to explore the pheno. char. captured by latent features",
                        # type=list,
                        default=[])

    parser.add_argument("--deseq2",
                        help="path to the file generated by the R script using DESeq2 to obtain the lof2FC values")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # declare input integration or not
    # set the h_int to True if you have more than one file to integrate
    if args.horizontal:
        data = horizontal_integration(args.input_dir, args.reference_column, args.file_format, args.separator)
    else:
        data = pd.read_csv(args.input_dir, args.reference_column, args.file_format, args.separator)

    # prefilter the input data
    filt_data = prefiltering_f(data)
    filt_data.set_index(data.columns[0], inplace=True)

    # prepare the data for vae application
    tr_data = filt_data.transpose()

    if args.minmax:
        tr_data = minmax_hive_f(args.input_dir, tr_data)
        tr_data = tr_data.rename(columns=(dict(zip(list(tr_data.columns), list(filt_data.index)))))

        # save MinMax scaled data
        tr_data.to_csv(f"{args.output_dir}minmax_scaled_hive_input_data.tsv", sep="\t")

    else:
        pass

    # encoder
    ei, el3a = setup_encoder_f(tr_data)
    sinput, ls = latent_space_f(ei, el3a, sampling_f)
    encoder, esumm = encoder_model_f(ei, sinput, ls)

    # decoder
    di, dl3a = setup_decoder_f()
    reco, decoder, dsumm = decoder_model_f(tr_data, di, dl3a)

    combine_structure = decoder(encoder(ei)[2])

    msle = msle_loss_f(tr_data, ei, combine_structure)
    kl_loss = kl_loss_f(sinput[0], sinput[1])
    vae_loss = vae_loss_f(msle, kl_loss)

    # Variational AutoEncoder initialization
    vae, vsumm = vae_init_f(ei, combine_structure, ls, vae_loss)

    # Latent Space of created VAE model
    fitting, pred, latent_space = vae_f(vae, tr_data)

    #### save LS
    latent_space.to_csv(f"{args.output_dir}latent_space.tsv", sep="\t")
    ####

    # predict reconstructed data for performance calculation
    pred_reco = decoder.predict(encoder.predict(tr_data)[2])

    # EVALUATION METRICS
    evaluation_met = {}
    # perplexity
    mean_perpl_score = perplexity_f(tr_data, pred_reco)
    evaluation_met["Mean perplexity"] = mean_perpl_score
    print("Mean perplexity score:", mean_perpl_score)

    # MSE
    mean_mserror = mse_f(tr_data, pred_reco)
    evaluation_met["MSE"] = mean_mserror
    print("Mean Square Error:", mean_mserror)

    # MAE
    mean_maerror = mae_f(tr_data, pred_reco)
    evaluation_met["MAE"] = mean_maerror
    print("Mean Absolute Error:", mean_maerror)

    # kl divergence
    kl = kl_divergence_f(tr_data, pred_reco)
    evaluation_met["KL-div"] = kl
    print("KL-divergence:", kl)

    # rmsle
    rmsle = rmsle_f(tf.constant(tr_data, dtype=tf.float32), tf.constant(pred_reco, dtype=tf.float32))
    evaluation_met["RMSLE"] = rmsle
    print("RMSLE:", rmsle)

    # loss plot
    loss_plot_f(fitting)

    # save evaluation metrics
    pd.DataFrame(evaluation_met).to_csv(f"{args.output_dir}HIVE_VAE_evaluation_metrics.tsv", sep="\t", index=False)

    print("VAE: Finished")
    print("START RFR")

    # REGRESSION WITH CROSS VALIDATION
    # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

    input_rfr = filt_low_expr_f(args.reference_column, data)

    # save Random Forest Regression input
    input_rfr.to_csv(f"{args.output_dir}NoLowExp_hive_rfr_input_data.tsv", sep="\t")

    latent_space_feat = latent_space[list(latent_space.columns)[1:]]
    # MinMax scale latent features to be in the same value-magnitude of the genes that will be regressed
    scaled_latent_space = minmax_hive_f(args.output_dir, latent_space_feat)
    smaller_class = np.min(list(Counter(args.rfr_classes).values()))
    rfr_output_dir = f"{args.output_dir}CV_{smaller_class}fold_classes/SHAP_per_fold/"
    if not os.path.exists(rfr_output_dir):
        os.makedirs(rfr_output_dir)

    mse = {}
    rmse = {}
    mae = {}
    r2 = {}
    maae = {}


    n_metrics = 5
    # this will results in a final output files containing the performance measures of rfr per fold per latent feature
    meta = {"Fold": ["1"] * n_metrics + ["2"] * n_metrics + ["3"] * n_metrics,
            "Measure": ["MSE", "RMSE", "MAE", "R2", "MAAE"] * smaller_class}

    skf = StratifiedKFold(n_splits=smaller_class, shuffle=True, random_state=42)
    reg_X = input_rfr

    if args.shap:
        for f in range(len(list(scaled_latent_space.columns))):
            print("Latent Feature", f)
            i = 0
            reg_y = scaled_latent_space[str(f)].values
            start = time()
            meta[f"LF{f + 1}"] = []

            best_rmse_index = None
            best_rmse_value = float('inf')

            for train_index, test_index in skf.split(reg_X, args.rfr_classes):
                print("fold", i)
                x_train = reg_X.iloc[train_index]
                x_test = reg_X.iloc[test_index]
                y_train = reg_y[train_index]
                y_test = reg_y[test_index]

                # Random Forest Regression
                rfr = RandomForestRegressor(n_estimators=100, random_state=42)

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
                meta[f"LF{f + 1}"] += [mse, rmse, mae, r2, maae]

                # SHAP importance calculation just on the best fold

                if rmse < best_rmse_value:
                    best_rmse_value = rmse
                    best_rmse_index = i

                    shap_ranked_mean = shap_f(x_test, rfr)
                    # use the below line and follow the plot-comments in shap_f function to plot the shap values
                    # shap_ranked_mean, plot = shap_f(scaled_latent_space, rfr)

                    print("SAVE SHAP scores...")
                    shap_ranked_mean.to_csv(f"{rfr_output_dir}/rfr_SHAP_ranked_mean_LF{f + 1}.tsv", sep="\t", index=False)

                print("DONE...")
                i += 1

            print("Overall LF time: ", (time() - start) / 60, "minutes")

        pd.DataFrame(meta).to_csv(f"{args.output_dir}/rfr_meta_infos.tsv",
                                  sep="\t",
                                  index=False)
        # GENE SELECTION

        all_bestf_shaps = []
        for f in range(1, len(latent_space.columns) + 1):
            file_shaps = pd.read_csv(f"{rfr_output_dir}/rfr_SHAP_ranked_mean_LF{f}.tsv", sep="\t")
            shap_values = file_shaps.mean_shap_value
            all_bestf_shaps += list(shap_values)

        selected_shap_neg = {}  # per feature gene shap values top negatives
        selected_shap_pos = {}  # and positives
        store_ranking = {}

        # create a dictionary of dictionaries where the first keys are the positions of genes and the values will become
        # the gene in that position for each feature associated to the corresponding shap value
        for p in range(len(reg_X)):
            store_ranking[f"pos{p + 1}"] = {}

        for f in range(1, len(latent_space.columns) + 1):
            f_shaps = pd.read_csv(f"{rfr_output_dir}/rfr_SHAP_ranked_mean_LF{f}.tsv",
                                  sep="\t")
            neg, pos = shap_bin_selection_f(60, all_bestf_shaps, f_shaps)  # change the number of bins

            selected_shap_neg[f] = list(
                f_shaps["ref"][f_shaps.mean_shap_value <= neg])  # all genes in the smaller bins below the neg shap mean
            selected_shap_pos[f] = list(
                f_shaps["ref"][f_shaps.mean_shap_value >= pos])  # all genes in the higher bins above the pos shap mean

            # GENE RANKING PER FEATURE
            sel_f = selected_shap_pos[f] + selected_shap_neg[f]
            shap_f_sel = f_shaps[f_shaps.gene_name.isin(sel_f)].copy()
            shap_f_sel["mean_shap_value"] = [abs(el) for el in list(shap_f_sel["mean_shap_value"])]
            shap_f_rank = shap_f_sel.sort_values(by="mean_shap_value", ascending=False, ignore_index=True)

            for p in range(len(shap_f_rank)):  # this is just a control for the real number of positions in the dataset
                # can happen that a gene has the same position in two or more features so always keep the higher value
                # for future ranking
                if shap_f_rank["gene_name"].iloc[p] not in list(store_ranking[f"pos{p + 1}"].keys()):
                    store_ranking[f"pos{p + 1}"][shap_f_rank["gene_name"].iloc[p]] = shap_f_rank["mean_shap_value"].iloc[p]

                else:
                    if store_ranking[f"pos{p + 1}"][shap_f_rank["gene_name"].iloc[p]] < shap_f_rank["mean_shap_value"].iloc[p]:
                        store_ranking[f"pos{p + 1}"][shap_f_rank["gene_name"].iloc[p]] = shap_f_rank["mean_shap_value"].iloc[p]
                    else:
                        pass

        # FINAL GENE SELECTION
        # for each position sort the absolute value of each gene and append to a single final list
        ranked_selected_genes = []
        for p in list(store_ranking.keys()):
            tmp_df = pd.DataFrame({"gene_name": store_ranking[p].keys(),
                                   "shap_abs_value": store_ranking[p].values()})
            tmp_df_rank = tmp_df.sort_values(by="shap_abs_value", ascending=False, ignore_index=True)
            for g in list(tmp_df_rank.gene_name):
                # if a gene is already present in the list means that in at least one features it already have a major
                # rank order so is not appended again
                if g not in ranked_selected_genes:
                    ranked_selected_genes.append(g)
                else:
                    pass

        # HIVE OUTPUT RANKED SELECTED GENES
        with open(f"{args.output_dir}HIVE_gene_selection_shap.txt", "w") as hive_out:
            for g in list(np.unique(ranked_selected_genes)):
                hive_out.write(f"{g}\n")

    elif args.gini:

        for f in range(len(list(scaled_latent_space.columns))):
            print("Latent Feature", f)
            i = 0
            reg_y = scaled_latent_space[str(f)].values
            start = time()
            meta[f"LF{f + 1}"] = []

            best_rmse_index = None
            best_rmse_value = float('inf')

            for train_index, test_index in skf.split(reg_X, args.rfr_classes):
                print("fold", i)
                x_train = reg_X.iloc[train_index]
                x_test = reg_X.iloc[test_index]
                y_train = reg_y[train_index]
                y_test = reg_y[test_index]

                # Random Forest Regression
                rfr = RandomForestRegressor(n_estimators=100, random_state=42)

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
                meta[f"LF{f + 1}"] += [mse, rmse, mae, r2, maae]

                # GINI importance calculation just on the best fold

                if rmse < best_rmse_value:
                    best_rmse_value = rmse
                    best_rmse_index = i

                    gini_importance = rfr.feature_importances_
                    print("SAVE GINI importance...")
                    gini_df = pd.DataFrame({"gene_name": input_rfr.columns,
                                            "GINI": gini_importance}).sort_values(by="GINI",
                                                                                  ascending=False,
                                                                                  ignore_index=True)

                    gini_df.to_csv(f"{rfr_output_dir}/rfr_GINI_LF{f + 1}.tsv", sep="\t", index=False)

                print("DONE...")
                i += 1

            print("Overall LF time: ", (time() - start) / 60, "minutes")

            pd.DataFrame(meta).to_csv(f"{args.output_dir}/rfr_meta_infos.tsv",
                                      sep="\t",
                                      index=False)
        # GENE SELECTION

        all_bestf_gini_sel = []
        store_ranking = {}
        for f in range(1, len(latent_space.columns) + 1):
            f_gini = pd.read_csv(f"{rfr_output_dir}/rfr_GINI_LF{f}.tsv", sep="\t")
            selection_gini100 = f_gini.head(101)
            all_bestf_gini_sel += list(selection_gini100.gene_name)

            selection_gini100["GINI"] = [abs(el) for el in list(selection_gini100["GINI"])]

            for p in range(len(selection_gini100)):
                # this is just a control for the real number of positions in the dataset
                # can happen that a gene has the same position in two or more features so always keep the higher value
                # for future ranking
                if selection_gini100["gene_name"].iloc[p] not in list(store_ranking[f"pos{p + 1}"].keys()):
                    store_ranking[f"pos{p + 1}"][selection_gini100["gene_name"].iloc[p]] = selection_gini100["GINI"].iloc[p]

                else:
                    if store_ranking[f"pos{p + 1}"][selection_gini100["gene_name"].iloc[p]] < selection_gini100["GINI"].iloc[p]:
                        store_ranking[f"pos{p + 1}"][selection_gini100["gene_name"].iloc[p]] = selection_gini100["GINI"].iloc[p]
                    else:
                        pass

        # FINAL GENE SELECTION
        # for each position sort the absolute value of each gene and append to a single final list
        ranked_selected_genes = []
        for p in list(store_ranking.keys()):
            tmp_df = pd.DataFrame({"gene_name": store_ranking[p].keys(),
                                   "shap_abs_value": store_ranking[p].values()})
            tmp_df_rank = tmp_df.sort_values(by="shap_abs_value", ascending=False, ignore_index=True)
            for g in list(tmp_df_rank.gene_name):
                # if a gene is already present in the list means that in at least one features it already have a major
                # rank order so is not appended again
                if g not in ranked_selected_genes:
                    ranked_selected_genes.append(g)
                else:
                    pass

        with open(f"{args.output_dir}HIVE_gene_selection_gini.txt", "w") as hive_out:
            for g in list(np.unique(ranked_selected_genes)):
                hive_out.write(f"{g}\n")

    if args.pheno:
        pheno_dir = args.output_dir + "phenotypic_char/"
        if not os.path.exists(pheno_dir):
            os.makedirs(pheno_dir)

        ls_pheno = latent_space.drop(latent_space[latent_space.ref.str.contains("ctrl")].index, inplace=True)
        if len(np.unique(args.pheno)) <= 2:
            # point biserial correlation
            i = 0
            stress_f_corr = {"LF": [], "p-value": [], "correlation": []}
            for f in list(latent_space.columns)[1:]:
                if scipy.stats.pointbiserialr(args.pheno, ls_pheno[f+1])[1] <= 0.01:
                    i += 1
                    stress_f_corr["LF"].append(f"LF{f}")
                    stress_f_corr["p-value"].append(scipy.stats.pointbiserialr(args.pheno, ls_pheno[f+1])[1])
                    stress_f_corr["correlation"].append(scipy.stats.pointbiserialr(args.pheno, ls_pheno[f+1])[0])

            stress_f_corr_df = pd.DataFrame(stress_f_corr).sort_values(by="p-value")
            stress_f_corr_df.to_csv(f"{pheno_dir}LFs_pbiser_correlation_sig_p001.tsv", sep="\t", index=False)

        else:
            ls_pheno["ref"] = args.pheno

            # Kruskal-Wallis test
            kw_dict = {}
            with open(f"{pheno_dir}LFs_kruskalwallis.txt", "w") as kw_out:
                for f in list(latent_space.columns)[1:]:
                    groups = [ls_pheno[f][ls_pheno.ref == g] for g in np.unique(args.pheno)]
                    kw = kruskal(*groups)
                    kw_out.write(f"Kruskal-Wallis LF{f}: {kw}\n")
                    kw_dict[f"LF{f}"] = kw[1] # for later usage

            # Dunns test
            dnn_dir = args.output_dir + "phenotypic_char/Dunns_test/"
            if not os.path.exists(dnn_dir):
                os.makedirs(dnn_dir)

            dnn_dict = {}
            for f in list(latent_space.columns)[1:]:
                dnn = scikit_posthocs.posthoc_dunn(ls_pheno, val_col=f, group_col="ref")
                with open(f"{dnn_dir}LF{f}_dunn_metrix.txt",  "w") as dunn_out:
                    dunn_out.write(dnn.to_string())
                dnn_dict[f"LF{f}"] = {}
                for i in range(len(dnn.index)):
                    for j in range(i + 1, len(dnn.columns)):
                        couple = dnn.index[i] + "/" + dnn.columns[j]
                        v = dnn.iloc[i, j]
                        dnn_dict[f"LF{f}"][couple] = v

                # -log(10) p-value transformation
                f_dnn_log = pd.DataFrame({f"LF{f}": list(dnn_dict[f"LF{f}"].keys()),
                                         "p-value": list(dnn_dict[f"LF{f}"].values())})

                f_dnn_log["-log10(p-value)"] = -np.log10(f_dnn_log["p-value"])

                f_dnn_log.to_csv(f"{dnn_dir}LF{f}_dunn_log_pval.tsv", sep="\t", index=True)

            # all Dunns' test in one dataframe
            dnn_df = pd.DataFrame(dnn_dict)
            dnn_df.to_csv(f"{pheno_dir}LFs_dunn.tsv", sep="\t")

            # combine results of both tests sorting LFs by KW p-values
            kw_sort = pd.DataFrame({"LFs": list(kw_dict.keys()),
                                    "p-value": list(kw_dict.values())}).sort_values(by="p-value")
            dnn_df[list(kw_sort.LFs)].to_csv(f"{pheno_dir}LFs_kw_dunn_correlation.tsv", sep='\t')

    if args.deseq2:
        condition_dir = args.output_dir + "gene_condition_assoc/"
        if not os.path.exists(condition_dir):
            os.makedirs(condition_dir)

        l2fc = pd.read_csv(args.deseq2, sep=",", index_col=0)
        l2fc = l2fc.fillna(0)
        l2fc_selected = l2fc[l2fc.ref.isin(ranked_selected_genes)]
        l2fc_abs = abs(l2fc_selected)

        inter_cond_diff = l2fc_abs.apply(calc_distance_f, axis=1)
        # reference condition is the condition for which each gene has the max l2fc value
        inter_cond_diff['ref_condition'] = l2fc_abs.idxmax(axis=1)

        # find threshold for gene condition association
        all_values = []
        dist = inter_cond_diff.copy()
        for index, row in inter_cond_diff.iterrows():
            ref_cond = row['ref_condition']
            other_vals = row.drop([ref_cond, 'ref_condition'])
            all_values.extend(other_vals)

            # distance columns

            columns_to_exclude = [ref_cond, "ref_condition"]
            dist.loc[index, columns_to_exclude] = np.nan

        min_val = min(all_values)
        max_val = max(all_values)
        mean_val = np.mean(all_values)
        sigmas_ori = np.std(all_values)
        sigmas = 1

        plt.figure(figsize=(10, 8))
        plt.hist(all_values, bins=100)
        plt.axvline(sigmas, color='red', linestyle='--', label='Sigma')
        plt.axvline(sigmas_ori, color='green', linestyle='--', label='Sigma_ori')

        plt.title('Distance distribution')
        plt.grid(True)
        plt.savefig(f"{condition_dir}inter_cond_diff_distro.pdf")

        dist['ref_condition'] = inter_cond_diff['ref_condition']

        # Group condition by pairs
        cols = l2fc.columns.tolist()
        pairs = list(itertools.combinations(cols, 2))
        couples = ['{}-{}'.format(pair[0], pair[1]) for pair in pairs]
        dist_couples = pd.DataFrame(index=dist.index, columns=couples)

        for index, row in dist.iterrows():
            ref_l2fc = row.iloc[-1]
            couples_with_ref = [couple for couple in couples if ref_l2fc in couple]

            for el in couples_with_ref:
                cond1, cond2 = el.split('-')
                if cond1 != ref_l2fc:
                    dist_couples.loc[index, el] = row[cond1]
                else:
                    dist_couples.loc[index, el] = row[cond2]

        # differences between pairs of conditions per gene
        dist_couples['ref_condition'] = inter_cond_diff['ref_condition']

        # filter out those differences not passing the threshold
        filtered_values = {}
        df_list = []

        i = 1
        for couple in couples:
            couple_df = dist_couples[[couple]]
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

        df_pathogens = pd.DataFrame(columns=l2fc.columns)

        for index, row in binarized.iterrows():
            pathogens_presence = [0] * len(df_pathogens.columns)

            for col in binarized.columns:
                pathogens = col.split('-')
                if row[col] == 1:
                    for pathogen in pathogens:
                        if pathogen in df_pathogens.columns:
                            pathogens_presence[df_pathogens.columns.get_loc(pathogen)] = 1

            df_pathogens.loc[index] = pathogens_presence

        outside_sigmas = dist.loc[dist.fillna(float('inf')).iloc[:, :-1].gt(sigmas).all(axis=1)]
        outside_sigmas_b = pd.DataFrame(index=outside_sigmas.index, columns=outside_sigmas.columns).fillna(0)
        l2fc_patho_spe = l2fc.loc[outside_sigmas_b.index]
        max_abs_column_index = l2fc_patho_spe.abs().idxmax(axis=1)
        patho_spe = pd.DataFrame(0, index=l2fc_patho_spe.index, columns=l2fc_patho_spe.columns)

        for i, col_name in enumerate(max_abs_column_index):
            patho_spe.loc[patho_spe.index[i], col_name] = 1

        df_conditions = pd.concat([df_pathogens, patho_spe])

        df_conditions['Associated_Conditions'] = df_conditions.apply(lambda row: df_conditions.columns[row == 1].tolist(),
                                                                     axis=1)

        df_conditions[['Associated_Conditions']].to_csv(f"{condition_dir}HIVE_gene_condition_assoc.tsv", sep="\t")

    else:
        pass

