library(S4Vectors)
library(stats4)
library(DESeq2)

## data
raw_data <- as.data.frame(read.delim("./data/peanut_data.csv", row.names=1))

## Peanut, select the condition ctrl and stressed
t_raw_data <- t(raw_data)
rownames(t_raw_data) <- c("ctrl", "ctrl",	"stress",	"stress",	"stress",	"stress",	"stress",	"stress",
                         "ctrl", "ctrl",	"stress",	"stress",	"stress",	"stress",	"stress",	"stress",
                         "ctrl", "ctrl",	"ctrl", "stress",	"stress",	"stress", "stress",	"stress",	"stress",
                         "ctrl", "stress", "ctrl", "ctrl",	"ctrl", "stress",	"stress",	"stress",
                         "ctrl", "stress")

## select the pathogene
indices_pathogen <- grep("^st", rownames(t_raw_data))

counts <- t_raw_data[indices_pathogen, ]
counts <- as.matrix(counts)

dds <- DESeqDataSetFromMatrix(countData = counts, colData = t_raw_data, design = rownames(t_raw_data))
# Condition = name of the condition column in metadata

rld <- rlog(counts)
rld_values <- assay(rld)

## LOG2FC
dds <- DESeq(dds)
res <- results(dds) # log2FC and pvalue

## save
write.csv(res, file = "./peanut_data_log2FC.tsv", row.names = TRUE)


