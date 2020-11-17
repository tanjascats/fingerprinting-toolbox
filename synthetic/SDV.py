import os

from sdv import SDV
from sdv import Metadata

import matplotlib.pyplot as plt
plt.style.use('classic')

import numpy as np
import pandas as pd
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

file_path=os.path.dirname(__file__)
traindata = os.path.join(file_path+'/',"traindata.csv")

def synth(data, output = False): 
    
    synthetic_data = os.path.join(file_path+'/',"SDV_syntraindata.csv")
    input_df = pd.read_csv(data, skipinitialspace=True)
    tables = {'ftable': input_df}
    
    trainjson = os.path.join(file_path, "meta.json")
    instance = os.path.join(file_path, "sdv.pkl")
    
    metadata = Metadata()
    metadata.add_table('ftable', data=tables['ftable'])
    metadata.to_json(trainjson)
    
    sdv = SDV()
    sdv.fit(metadata, tables)
    sdv.save(instance)
    
    sdv = SDV.load(instance)
    samples = sdv.sample_all(106)  #here: number of synthetic samples
    df=samples['ftable']
    df=df[input_df.columns]
    df.to_csv(synthetic_data, index=False)
    
    if output:
        
        syn_df = pd.read_csv(synthetic_data)
        
        # Label Encoding
        header = list(input_df.columns.values)
        
        for i in range(len(header)):
            
            if header[i] in ['Gender']:
                input_df.iloc[:,i]=le.fit_transform(list(input_df.iloc[:,i])).astype(np.float)
                syn_df.iloc[:,i]=le.fit_transform(list(syn_df.iloc[:,i])).astype(np.float)
        
        header = list(input_df.columns.values)
        correlations_syn = syn_df.corr()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cax = ax.matshow(correlations_syn, cmap='seismic', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(header),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(header)
        ax.set_yticklabels(header)
        plt.savefig('Plots/heatsyn.png')
        # plt.show()
        
        correlations_real = input_df.corr()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        cax = ax.matshow(correlations_real, cmap='seismic', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(header),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(header)
        ax.set_yticklabels(header)
        plt.savefig('Plots/heatreal.png')
        # plt.show()
        
        for i in range(len(header)):
            plt.hist(input_df.iloc[:,i], color='green', label='Real: '+header[i])
            plt.legend(fontsize='x-small')
            plt.hist(syn_df.iloc[:,i],histtype='step', color='blue', label='Syn: '+header[i])
            plt.legend(fontsize='x-small')
            plt.savefig('Plots/Histograms/'+header[i]+'.png')
            # plt.show()
        
if __name__ == '__main__': 
    
    synth(traindata, True)