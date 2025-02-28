# General imports
import argparse
import os
import random
import matplotlib
import re
import pandas as pd
import numpy as np

# Input scaling imports
from sklearn.preprocessing import MinMaxScaler

# VAE imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import mse
import scipy
from scipy.stats import entropy

# Clustering imports
from sklearn.cluster import KMeans
from sklearn import metrics


tf.random.set_seed(779)

def horizontal_integration(input_dir, reference_column, file_fmt, sep, header=0):
    """Merge each file with the next one maintaining the raw order (typically gene order) of the first file.
    The first column of all files has to contain the gene names.

    Parameters
    ----------
    input_dir : path to directory containing only data to be horizontally integrated
    reference_column : the reference column in which are stored the gene/molecules names on which integrate
    suff : file format of all files containing data to be integrated
    sep : separator character to correcly parse the files
    header : indicate the row that has to be considered as column_names, default=0

    Returns
    -------
    df : an horizontally integrated dataset
    """

    file_list = []
    df_list = []
    meta_info = {"sample_ref": [], "batch": []}
    for el in os.listdir(input_dir):
        if el.endswith(file_fmt):
            file_list.append(el)
            df_list.append(pd.read_csv(os.path.join(input_dir, el), sep=sep, header=header))
            print(el)
            print(pd.read_csv(os.path.join(input_dir, el), sep=sep, header=header))
    df = df_list[0]
    meta_info["sample_ref"] = list(df.columns[1:])
    meta_info["batch"] = list([0]*len(df.columns[1:]))
    for i, d in enumerate(df_list[1:]):
        df = df.merge(d, on=reference_column)
        
        meta_info["sample_ref"] += list(d.columns[1:])
        meta_info["batch"] += list([i+1]*len(d.columns[1:]))       
    
    meta_info_df = pd.DataFrame(meta_info) 
    return df, meta_info_df


def prefiltering_f(data):
    """Filter out all rows with variance below first distribution quartile

    Parameters
    ----------
    data : the integrated dataset

    Returns
    -------
    to_retain : dataframe containing genes to be retained for the analysis
    """

    data_no_zeroes = data

    variances = {}
    for r in range(len(data_no_zeroes)):
        arr = np.array(data_no_zeroes.iloc[r])[1:]
        var = np.var(arr)
        variances[data_no_zeroes[data_no_zeroes.columns[0]].iloc[r]] = var
    variances_df = pd.DataFrame({"ref": list(variances.keys()),
                                 "variance": list(variances.values())})
    thr_var = float(variances_df["variance"].quantile(0.25, interpolation='lower'))

    var_to_retain = variances_df[variances_df["variance"] > thr_var]

    to_retain = data_no_zeroes[data_no_zeroes[data_no_zeroes.columns[0]].isin(list(var_to_retain.ref))]

    return to_retain


def minmax_hive_f(data):
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

def seed_everything(seed=779):
    """"Seed everything
    """
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def sampling_f(inputs):
    """Sample the latent space from the latent distribution
    """
    ld_mean, ld_log_var = inputs
    batch = keras.backend.shape(ld_mean)[0]
    dim = keras.backend.int_shape(ld_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim), seed=779)

    z = ld_mean + keras.backend.exp(0.5 * ld_log_var) * epsilon

    return z


def setup_encoder_f(data, el1=5000, el2=1000, el3=500, kernel="lecun_normal", ls_size=80):
    """ENCODER SETUP - implemented to have 3 layers = [el1, el2, el3] and a latent space size of 80 latent features
    the latent space is sampled from the latent disribution (because of the autoencoder type)

    Parameters
    ----------
    data : min max scaled horizontal integrated dataset to be used as input for HIVE VAE
    el1 : number of nodes in the first layer of encoder, default=5000
    el2 : number of nodes in the second layer of encoder, default=1000
    el3 : number of nodes in the third layer of encoder, default=500
    kernel : distribution, default="lecun_normal"
    ls_size : number of latent features to be extracted from the latent distribution, default=80

    Returns
    -------
    encoder_input : input of the neural network that goes into the first layer
    encoder_l3_activation : latest activation function of the encoder
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
    """LATENT SPACE - uses the output of the encoder_setup_f function to retrieve the latent space 

    Parameters
    ----------
    encoder_input : input of the neural network that goes into the first layer
    encoder_l3_activation : latest activation function of the encoder
    sampling : sampling function (sampling_f) that defines how to sample from the latent distribution
    ls_size : number of latent features to be extracted from the latent distribution, default=80

    Returns
    -------
    sampling_inputs : parameters for the sampling function
    ls : latent space vector
    """

    ld_mean = layers.Dense(ls_size, name="ld_mean")(encoder_l3_activation)
    ld_log_var = layers.Dense(ls_size, name="ld_log_var")(encoder_l3_activation)
    sampling_inputs = [ld_mean, ld_log_var]

    ls = layers.Lambda(sampling, output_shape=(ls_size,), name="ls")(sampling_inputs)  # latent space vector

    return sampling_inputs, ls


def encoder_model_f(encoder_input, sampling_inputs, ls):
    """ENCODER - build-up the encorder model

    Parameters
    ----------
    encoder_input : input of the neural network that goes into the first layer
    sampling_inputs : sampling function (sampling_f) that defines how to sample from the latent distribution
    ls : latent space vector

    Returns
    -------
    encoder_model : the actual encoder model 
    # encoder_model_summary() : summary of the encoder model structure
    """
    encoder_model = keras.Model(inputs=encoder_input, outputs=sampling_inputs + [ls], name="encoder")

    # uncomment the last return to have a summary of the model
    return encoder_model, encoder_model.summary()


def setup_decoder_f(ls_size=80, dl1=500, dl2=1000, dl3=5000, kernel="lecun_normal"):
    """DECODER SETUP - has the same and inverse structure of the encoder, thus, 3 layers = [dl1, dl2, dl3]

    Parameters
    ----------
    ls_size : number of latent features to be extracted from the latent distribution, default=80
    dl1 : number of nodes in the first layer of decoder, default=500
    dl2 : number of nodes in the second layer of decoder, default=1000
    dl3 : number of nodes in the third layer of decoder, default=5000
    kernel : distribution, default="lecun_normal"
    
    Returns
    -------
    decoder_input : input of the neural network that goes into the first layer
    decoder_l3_activation : latest activation function of the decoder
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
    """DECODER - build-up the decorder model
    
    Parameters
    ----------
    data : min max scaled horizontal integrated dataset to be used as input for HIVE VAE reconstruction
    decoder_input : input of the neural network that goes into the first layer
    decoder_l3_activation : latest activation function of the decoder
    
    Returns
    -------
    reconstruction : predicted values from the DECODER reconstruction 
    decoder_model : the actual decoder model
    # decoder_model.summary() : summary of the decoder model structure
    """
    reconstruction = layers.Dense(data.shape[1])(decoder_l3_activation)  # reconstruction layer (final obj of decoder)
    decoder_model = keras.Model(decoder_input, reconstruction, name="decoder")
    # uncomment the last return to have a summary of the model
    return reconstruction, decoder_model, decoder_model.summary()


def msle_loss_f(data, encoder_input, outputs):
    """Calculate the MSLError loss
    
    Parameters
    ----------
    data : min max scaled horizontal integrated dataset to be used as input for HIVE VAE
    encoder_input : input of the neural network that goes into the first layer
    outputs : reconstruction data
 
    Returns
    -------
    msle : MSLE loss for the VAE evaluation
    """
    msle = keras.losses.MSLE(encoder_input, outputs[0])
    msle *= data.shape[1]
    return msle


def kl_loss_f(ld_mean, ld_log_var, kl_loss_weight=0.5):
    """Calculate the Kullback-Leibler loss
    
    Parameters
    ----------
    ld_mean : mean of the latent distribution
    ld_log_var : standard deviation of the latent distribution
    kl_loss_weight : wheight to be taken in the calculation of the kl loss, default=0.5
 
    Returns
    -------
    kl_loss : KL loss for the VAE evaluation
    """
    kl_loss = 1 + ld_log_var - keras.backend.square(ld_mean) - keras.backend.exp(ld_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -kl_loss_weight
    return kl_loss


def vae_loss_f(msle, kl_loss):
    """Calculate the total VAE loss

    Parameters
    ----------
    msle : MSLE loss for the VAE evaluation
    kl_loss : KL loss for the VAE evaluation

    Returns
    -------
    vae_loss : total loss for VAE input reconstruction
    """
    vae_loss = keras.backend.mean(msle + kl_loss)
    return vae_loss


def vae_init_f(encoder_input, outputs, ls, vae_loss):
    """Initialization of the VAE model using encoder and decoder defined before

    Parameters
    ----------
    encoder_input : input of the neural network that goes into the first layer
    outputs : reconstruction data
    ls : latent space vector
    vae_loss : total loss for VAE input reconstruction

    Returns
    -------
    vae_model: initialized VAE model
    # vae_model_summary() : summary of the VAE model structure
    """
    vae_model = keras.Model(encoder_input, [outputs, ls], name="vae")
    vae_model.add_loss(vae_loss)
    vae_model.compile(keras.optimizers.Adam())
    # uncomment the last return to have a summary of the model
    return vae_model, vae_model.summary()


def vae_f(vae, data, epochs=300, batch_size=32):
    """Fitting the VAE on the data

    Parameters
    ----------
    vae : the VAE model
    data : min max scaled horizontal integrated dataset to be used as input for HIVE VAE
    epochs : number of epochs on which to train the VAE, default=300
    batch_size : batch size to consider during training, default=32

    Returns
    -------
    vae_fitting : history of losses per epoch 
    predictions : predictions after the decoder application
    latent_space : actual latent space to be used for further analysis
    """
    vae_fitting = vae.fit(data, epochs=epochs, batch_size=batch_size)
    predictions, latent_space = vae.predict(data)
    labels = np.array(data.index)
    latent_space = pd.DataFrame(latent_space).set_index(labels)  # the output needed for further analyses
    return vae_fitting, predictions, latent_space


def perplexity_f(y_true, y_pred):
    """Evaluate the fitted VAE with perplexity score

    Parameters
    ----------
    y_true : real data in input
    y_pred : predicted data after VAE application 

    Returns 
    -------
    np.mean(perpl, axis=0) : mean value of perplexity scores across epochs
    """
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
    """Calculates the Mean Squared Error of the VAE prediction

    Parameters 
    ----------
    y_true : real data in input
    y_pred : predicted data after VAE application 

    Returns 
    -------
    mse_score : MSE score
    """
    # mse = keras.losses.MeanSquaredError()
    mse_score = np.mean(np.square(y_true - y_pred), axis=0)
    return  mse_score # np.mean(mse_score, axis=0)


def mae_f(y_true, y_pred):
    """Calculates the Mean Absolute Error of the VAE prediction

    Parameters 
    ----------
    y_true : real data in input
    y_pred : predicted data after VAE application 

    Returns 
    -------
    mae_score : MAE score
    """
    mae = keras.losses.MeanAbsoluteError()
    mae_score = np.square(mae(y_true, y_pred))
    return mae_score # np.mean(mae_score, axis=0)


def clustering_metrics_f(ls_values, batches):
    """Calculates 6 different clustering metricies to evaluate the batch effect reduction in the latent space

    Parameters 
    ----------
    ls_values : list of lists representing the values of each latent feature in the latent space along samples
    batches : list of samples belonging batches, see this as the ground truth to be compared with the clustering results
   
    Returns 
    -------
    cluster_metrics : dictionary of metrics names as keys() and list of values per metric per number of cluster considered, as values() 
    """

    cluster_metrics = {"n_clusters": list(range(2, 11)), "homogeneity score": [], "completeness score": [], "v_measure": [], "adj rand score": [],
                        "adj mutual score": [], "jaccard score": [], "silhouette_score": [], "inertia": []}

    for i in range(2, 11):
        kmeans = list(KMeans(n_clusters=i, random_state=42, n_init=4).fit_predict(ls_values))
        kmeans_elb = KMeans(n_clusters=i, random_state=42, n_init=4).fit(ls_values)
    
        cluster_metrics["homogeneity score"].append(metrics.homogeneity_score(batches, list(kmeans)))
        cluster_metrics["completeness score"].append(metrics.completeness_score(batches, list(kmeans)))
        cluster_metrics["v_measure"].append(metrics.v_measure_score(batches, list(kmeans)))
        cluster_metrics["adj rand score"].append(metrics.adjusted_rand_score(batches, list(kmeans)))
        cluster_metrics["adj mutual score"].append(metrics.adjusted_mutual_info_score(batches, list(kmeans)))
        cluster_metrics["jaccard score"].append(metrics.jaccard_score(batches, list(kmeans), average="micro"))
        cluster_metrics["silhouette_score"].append(metrics.silhouette_score(ls_values, kmeans, metric="braycurtis"))
        cluster_metrics["inertia"].append(kmeans_elb.inertia_)
    
    return cluster_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="%(prog)s [options]",
                                     description="",
                                     epilog="")

    parser.add_argument("-i", "--input_dir",
                        help="input directory in which to find the raw expression data for HIVE application, "
                             "e.g. /home/HIVE_input_data/",
                        default="./")

    parser.add_argument("-o", "--output_dir",
                        help="output directory in which to save HIVE output files",
                        default="./yHIVE_results")

    parser.add_argument("-hz", "--horizontal",
                        help="indicate if an horizontal integration of two or more dataset is needed",
                        action="store_true")
    
    parser.add_argument("-rc", "--reference_column",
                        help="indicate the name of the column containing the gene names",
                        type=str,
                        default="ref")

    parser.add_argument("-ff", "--file_format",
                        help="indicate the input file/s extension",
                        type=str,
                        default=".csv")

    parser.add_argument("-sep", "--separator",
                        help="indicate the separator character for columns in input data",
                        type=str,
                        default=",")

    parser.add_argument("--minmax",
                        help="indicate whether to apply or not the MinMaxScaler [0,1] range",
                        action="store_true",
                        default=True)

    parser.add_argument("-ep", "--epochs",
                        help="indicate the number of epochs for the VAE training",
                        type=int,
                        default=300)

    parser.add_argument("-en", "--encoder",
                        help="indicate the number of nodes for each of the 3 layers of the NN-encoder, e.g. 5000 1000 500",
                        nargs="+",
                        type=int,
                        default=[5000, 1000, 500])
    
    parser.add_argument("-de", "--decoder",
                        help="indicate the number of nodes for each of the 3 layers of the NN-decoder, e.g. 500 1000 5000",
                        nargs="+",
                        type=int,
                        default=[500, 1000, 5000])
    
    parser.add_argument("-b", "--batches",
                        help="multi-hot encoding vector in which you assign the same value to samples coming from the same batch, e.g. 0 0 0 1 1 1 2 2",
                        nargs="+",
                        type=int,
                        default=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2])

    args = parser.parse_args()
    
    seed_everything()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # declare input integration or not # to be corrected 
    # set the h_int to True if you have more than one file to integrate
    if args.horizontal:
        print(args.input_dir)
        data, meta_info = horizontal_integration(args.input_dir, str(args.reference_column), str(args.file_format), str(args.separator))
        
        meta_info.to_csv(f"{args.output_dir}yHIVE_input_data_batches.tsv", sep="\t", index=False)
        filt_data = data.copy() 
        # prefilter the input data  
        filt_data = prefiltering_f(data)
        filt_data.set_index(data.columns[0], inplace=True)
        filt_data.to_csv(f"{args.output_dir}yHIVE_filtered_input_data.tsv", sep="\t")
    else:
        for el in os.listdir(args.input_dir):
            if el.endswith(str(args.file_format)):
                data = pd.read_csv(f"{args.input_dir}/{el}", sep=str(args.separator))
                filt_data = prefiltering_f(data)
                filt_data.set_index(data.columns[0], inplace=True)
                filt_data.to_csv(f"{args.output_dir}yHIVE_filtered_input_data.tsv", sep="\t")

    # prepare the data for vae application
    if args.minmax:
        mm_data = minmax_hive_f(filt_data)
        # save MinMax scaled data
        mm_data.to_csv(f"{args.output_dir}yHIVE_minmax_scaled_input_data.tsv", sep="\t")
    else:
        pass

    #############
    #### VAE ####
    #############
    print("HIVE computing VAE (1/2)")
    # VAE input 
    tr_data = mm_data.transpose()
    
    # encoder
    ei, el3a = setup_encoder_f(tr_data, el1=list(args.encoder)[0], el2=list(args.encoder)[1], el3=list(args.encoder)[2])
    sinput, ls = latent_space_f(ei, el3a, sampling_f)
    encoder, esumm = encoder_model_f(ei, sinput, ls)

    # decoder
    di, dl3a = setup_decoder_f(dl1=list(args.decoder)[0], dl2=list(args.decoder)[1], dl3=list(args.decoder)[2])
    reco, decoder, dsumm = decoder_model_f(tr_data, di, dl3a)

    combine_structure = decoder(encoder(ei)[2])

    msle = msle_loss_f(tr_data, ei, combine_structure)
    kl_loss = kl_loss_f(sinput[0], sinput[1])[0]
    vae_loss = vae_loss_f(msle, kl_loss)

    # Variational AutoEncoder initialization
    vae, vsumm = vae_init_f(ei, combine_structure, ls, vae_loss)

    # Latent Space of created VAE model
    fitting, pred, latent_space = vae_f(vae, tr_data)
   
    #### save LS
    latent_space.to_csv(f"{args.output_dir}yHIVE_latent_space.tsv", sep="\t")
    ####

    # predict reconstructed data for performance calculation
    pred_reco = decoder.predict(encoder.predict(tr_data)[2])
    
    # EVALUATION METRICS
    evaluation_met = {}
    
    # perplexity
    mean_perpl_score = perplexity_f(tr_data, pred_reco)
    evaluation_met["Mean perplexity"] = mean_perpl_score

    # MSE
    mean_mserror = mse_f(tr_data, pred_reco)
    evaluation_met["MSE"] = pd.DataFrame(mean_mserror)[0].mean(axis=0)

    # MAE
    mean_maerror = mae_f(tr_data, pred_reco)
    evaluation_met["MAE"] = mean_maerror

    #### Save VAE evaluation metrics
    pd.DataFrame({"Epoch": list(range(args.epochs)),
                  "Loss": fitting.history["loss"]}).to_csv(f"{args.output_dir}yHIVE_VAE_loss_per_epoch.tsv", sep="\t", index=False)

    pd.DataFrame({"Metric": evaluation_met.keys(),
                  "Metric_value": evaluation_met.values()}).to_csv(f"{args.output_dir}yHIVE_VAE_evaluation_metrics.tsv", sep="\t", index=False)
    ####

    ####################
    #### CLUSTERING ####
    ####################
    if args.batches:    
        print("HIVE evaluate batch-effect reduction (2/2)")
        
        ls_values = [list(a) for a in latent_space.values]

        cluster_metrics = clustering_metrics_f(ls_values, args.batches)
        cluster_metrics_df = pd.DataFrame(cluster_metrics)
        cluster_metrics_df.set_index("n_clusters", inplace=True)

        #### Save batch-effect reduction evaluation metrics
        cluster_metrics_df.T.to_csv(f"{args.output_dir}yHIVE_batch_eff_red_evaluation_metrics.tsv", sep="\t")
        ####
    else:
        print("If you want to evaluate BATCH-EFFECT reduction, please provide the -b argument")

    






















