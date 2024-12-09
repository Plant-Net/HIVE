# HIVE
HIVE is a novel tool to perform unpaired multi-transcriptomics data integration and to retrieve important features acting during plant-stress response. 

Want to take a look to our manuscript first? Here it is: https://doi.org/10.1101/2024.03.04.583290.

# HIVE workflow
![fig1_censored](https://github.com/user-attachments/assets/53dbb2b0-f696-4109-b61f-4290964a00a1)


# Scripts 
As in the HIVE workflow, here we furnish 3 different scripts named according to the color of the "HIVEmetropolitan-lines" they refer their analysis to: yHIVE1.0.py for the yellow-line, bHIVE1.0.py for the blue-line, and gHIVE1.0.py for the green-line.

## yHIVE 
This script will perform horizontal integrazion of data (if required), implement and evaluate a Variational AutoEncoder and batch-effect reduction.

### System requirements
It requires at least 8 GB of free memory space to run correctly.

### Python requirements
It is implemented and tested on python 3.10.9.
Important libraries and versions are:
- tensorflow v2.11.0
- sklearn v1.2.2
- numpy v1.24.2
  
Please install the remaining imports that can be missing, if not already done: ```pandas, scipy```

### Usage
```python yHIVE1.0.py -h``` will give you the following help message:

```
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        input directory in which to find the raw expression data for HIVE application, e.g. /home/HIVE_input_data/
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory in which to save HIVE output files, e.g. /home/yHIVE_results/
  -hz, --horizontal     indicate if an horizontal integration of two or more dataset is needed
  -rc REFERENCE_COLUMN, --reference_column REFERENCE_COLUMN
                        indicate the name of the column containing the gene names
  -ff FILE_FORMAT, --file_format FILE_FORMAT
                        indicate the input file/s extension
  -sep SEPARATOR, --separator SEPARATOR
                        indicate the separator character for columns in input data
  --minmax              indicate wether to apply or not the MinMaxScaler [0,1] range
  -ep EPOCHS, --epochs EPOCHS
                        indicate the number of epochs for the VAE training
  -en ENCODER [ENCODER ...], --encoder ENCODER [ENCODER ...]
                        indicate the number of nodes for each of the 3 layers of the NN-encoder
  -de DECODER [DECODER ...], --decoder DECODER [DECODER ...]
                        indicate the number of nodes for each of the 3 layers of the NN-decoder
  -b BATCHES [BATCHES ...], --batches BATCHES [BATCHES ...]
                        multi-hot encoding vector in which you assign the same value to samples coming from the same batch
```
For a demo trial of yHIVE you can find in the ```./yHIVE_demo/data/``` folder a possible usecase from our study, please run the following commands replacing the parameters accordingly, if needed:
```
python yHIVE1.0.py -i ./yHIVE_demo/data/ -o --minmax
```
or, with a specific case with the need for horizontal integration of data
```
python yHIVE1.0.py --hz -i /path/to/input/folder/ -o /path/to/output/folder -rc "name_of_ref_column" -ff ".fileformat" -sep "separator_character" --minmax -b 0 1 2 4 
```
The batch parameter ```-b``` allows you to assign a label to each sample coming form the same experiment thus its length should be the same as the number of samples.
If no horizontal integration is needed (e.g. you already have the intefrated data), just remove the ```-hz``` argument at the beginning of the command.
**If you need to modify the number of epochs or the number of neurons in each of the three layers of our implementation of VAE, use the arguments ```-ep```, ```-en```, ```-de```** like reported in the hepling message.

## gHIVE 
This script will perform the extraction of genes mostly related to the plant-stress response and provides the association to condition/s for each of the extracted genes.

### Python requirements
It is implemented and tested on python 3.10.9.
Important libraries and versions are:
- shap (tested on v0.46.0)
- sklearn v1.2.2
- itertools
  
Please install the remaining imports that can be missing, if not already done: ```pandas, numpy, matplotlib, collections, time```

### Usage 
```python gHIVE1.0.py -h``` will give you the following help message:

```
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory in which to save HIVE output files, e.g. /home/HIVE_results/
  -f FILTERED_DATA, --filtered_data FILTERED_DATA
                        path to the file of filtered read counts
  -mm MINMAX_DATA, --minmax_data MINMAX_DATA
                        path to the file of minmax scaled read counts
  -ls LATENT_SPACE, --latent_space LATENT_SPACE
                        path to the file regarding the latent space previously obtained
  -c RFR_CLASSES [RFR_CLASSES ...], --rfr_classes RFR_CLASSES [RFR_CLASSES ...]
                        indicate the vector corresponding to classes in your experimental design for the random forest regression step, MAINTAIN THE SAMPLE ORDER
  --gini                indicate if you want to use GINI as feature importance calculation for HIVE gene selection
  -fc FOLD_CHANGE, --fold_change FOLD_CHANGE
                        path to log2FoldChange file of HIVE selected genes
  -M, --merge_cols      indicate wether there is the need to merge columns of l2fc matrix for final molecule association to condition, e.g. replicates
  -cM COLS_TO_MERGE [COLS_TO_MERGE ...], --cols_to_merge COLS_TO_MERGE [COLS_TO_MERGE ...]
                        list of lists of names of columns to be merged
  -fM FINAL_COLS [FINAL_COLS ...], --final_cols FINAL_COLS [FINAL_COLS ...]
                        list of names of final merged columns
```
For a demo trial of gHIVE you can find possible usecase files in the ```./gHIVE_demo/``` folder. Please run the following commands replacing the parameters accordingly, if needed:
```
python gHIVE1.0.py -f ./gHIVE_demo/yHIVE_filtered_input_data.tsv -mm ./gHIVE_demo/yHIVE_minmax_scaled_input_data.tsv -ls ./gHIVE_demo/yHIVE_latent_space.tsv -fc ./gHIVE_demo/fold_change.tsv -M -cM 'as3' 'as6' 'as9' -cM 'ad3' 'ad6' 'ad9' -fM 'as' 'ad'
```
In general ```gHIVE1.0.py``` takes in input results form ```yHIVE1.0.py``` like ```filtered_input_data.tsv```, ```minmax_scaled_input_data.tsv```, and ```latent_space.tsv```. The latest arguments are needed only in the case you have to merge some conditions when calculating the association of the genes with the conditions. In this case we have the biotic stress splitted into 3 time points but in this phase their association can be considered as a whole so we merged the three time points of *A. stenosperma* (```'as3' 'as6' 'as9'```) into a single column (```'as'```) and the other three from *A. duranensis* (```'ad3' 'ad6' 'ad9'```) into the other (```'ad'```). 

**Please notice that the argument ```-cM``` has to be used multiple times according to the numebr of merged columns you want to create**. 

If there is no need to merge conditions, simply remove the latest 3 arguments from the command line, thus:
```
python gHIVE1.0.py -f ./gHIVE_demo/yHIVE_filtered_input_data.tsv -mm ./gHIVE_demo/yHIVE_minmax_scaled_input_data.tsv -ls ./gHIVE_demo/yHIVE_latent_space.tsv -fc ./gHIVE_demo/fold_change.tsv
```

This script automatically use SHAP score to assign an importance score to each gene during the random forest regression. If you want to perform a more classical GINI index the script can be called with the argument ```--gini```.
```
python gHIVE1.0.py -f ./gHIVE_demo/yHIVE_filtered_input_data.tsv -mm ./gHIVE_demo/yHIVE_minmax_scaled_input_data.tsv -ls ./gHIVE_demo/yHIVE_latent_space.tsv -fc ./gHIVE_demo/fold_change.tsv --gini
```

## bHIVE 
This script will compute the correlation of the Latent Features to the phenotypic conditions. Depending on the number of phenotype you want to correlate with the Latent Features (excluding the controls), it will either perform a point-biserial correlation or a Kruskal-Wallis test followed by the Dunn's test.

### Python requirements
It is implemented and tested on python 3.10.9.
Important libraries and versions are:
- scikit_posthocs (tested on v0.11.1)

Please install the remaining imports that can be missing, if not already done: ```pandas, numpy, math, scipy, sklearn```

### Usage 
```python bHIVE1.0.py -h``` will give you the following help message:
```
 -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory in which to save HIVE output files, e.g. /home/HIVE_results/
  -ls LATENT_SPACE, --latent_space LATENT_SPACE
                        path to the latent space file previously obtained with the yHIVE1.0.py script
  --pheno PHENO [PHENO ...]
                        indicate a numeric vector with numbers assigned to each sample (DO NOT CONSIDER CONTROLS)corresponding to phenotypic characteristics of interest, following their original order, to explore the pheno.
                        char. captured by latent features
```
For a demo trial of bHIVE you can find a possible usecase files in the ```./bHVE_demo/``` folder. Please run the following commands replacing the parameters accordingly, if needed:

With 2 phenotypes (e.g. ```bio``` = biotic stress, and ```abio``` = abiotic stress) it will perform a point-biserial correlation:
```
python bHIVE1.0.py -ls ./bHIVE_demo/yHIVE_latent_space.tsv --pheno ctrl ctrl bio bio bio bio bio bio ctrl ctrl bio bio bio bio bio bio ctrl ctrl ctrl abio abio abio abio abio abio ctrl abio ctrl ctrl ctrl abio abio abio ctrl abio
```

With 3 or more phenotypes (e.g. ```bio``` = biotic stress, ```abio``` = abiotic stress, and ```bioabio``` = cross stress) it will perform Kruskal-Wallis test before and then Dunn's test:
```
python bHIVE1.0.py -ls ./bHIVE_demo/yHIVE_latent_space.tsv --pheno ctrl ctrl bio bio bio bio bio bio ctrl ctrl bio bio bio bio bio bio ctrl ctrl ctrl bioabio bioabio bioabio abio abio abio ctrl abio ctrl ctrl ctrl abio abio abio ctrl abio 
```
**Please notice that the ```--pheno``` argument refers to the samples associated to a specific phenotype of interest, not directly to the Latent Features.**

## In any case, thanks for using HIVE! [The SMILE-team]
