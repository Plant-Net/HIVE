# HIVE
HIVE is a novel tool to perform unpaired multi-transcriptomics data integration and to retrieve important features acting during plant-stress response. 

# HIVE workflow
![fig1_censored](https://github.com/user-attachments/assets/53dbb2b0-f696-4109-b61f-4290964a00a1)


# Scripts 
As in the HIVE workflow, here we furnish 3 different scripts named according to the color of the "HIVEmetropolitan-lines" they refer their analysis to: yHIVE1.0.py for the yellow-line, bHIVE1.0.py for the blue-line, and gHIVE1.0.py for the green-line.

# yHIVE 
This script will perform horizontal integrazion of data (if required), implement and evaluate a Variational AutoEncoder

## System requirements
It requires at least 8 GB of free memory space to run correctly.
It is implemented and tested on python 3.10.9.
Important library version are:
- tensorflow v2.11.0
- sklearn v1.2.2
- numpy v1.24.2
Please install the remaining imports at your wish

## Usage
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
To obtain the latent space please for our usecase, please run the following command replacing the parameters accordingly (this step can take a while):
e.g.
```
python yHIVE1.0.py -i /path/to/input_dir/ -o /path/to/output_dir --minmax 
```

