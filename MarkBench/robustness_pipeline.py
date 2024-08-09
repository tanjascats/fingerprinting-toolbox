### This file performs the robustness pipeline for a given given dataset

## expected input: scheme with defined parameters (hyperparameters); output metric
## expected output: graphs for each attack (x-axis: attack strength; y-axis: robustness metric(e.g. false miss)


### internally: cash the fingerprinted data (10 times) -- use in the calculations and remove later
###             record runtime for extraction and insertions

## todo: optimisations, e.g. bit-flipping attack
