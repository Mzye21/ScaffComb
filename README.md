# ScaffComb: A Phenotype-Based Framework for Drug Combination Virtual Screening in Large-scale Chemical Datasets
This repository contains the source codes, the example data and some analysis codes.

## ScaffComb
![image](https://github.com/Mzye21/ScaffComb/blob/e78391d1dd0efb651ab030d0f90e0ab21a6b697e/Figures/scaffcomb.png)

## Dependences
* python 3.6.8
* tensorflow-gpu 1.12.0
* tensorboardx 2.0
* pytorch 1.3.1
* numpy 1.17.0
* rdkit 2020.03.1b1.0
* scipy 1.2.1
* scikit-learn 0.21.2
* matplotlib 3.1.1

## Usage
### Training GSG
    from GSG import AttnDGenerator
    dt_L1000 = pickle.load(open('data/data_GSG_input_train_example.pkl','rb'))
    dt_chembl = pickle.load(open('data/data_GSG_input_pretrain_example.pkl', 'rb'))
    tokens = pickle.load(open('data/tokens.pkl','rb'))
    # Pre-training with chembl scaffolds
    Trainer0 = AttnDGenerator(tokens, dt_chembl, batch_size=128, lr = 0.0001,
                               device=device, path='rst/pretrained/')
    Trainer0.train(1000)
    # Train with L1000 scaffolds
    Trainer1 = AttnDGenerator(tokens, dt_L1000, batch_size=128, lr = 0.0001,
                               device=device, path='rst/trained/')
    Trainer1.load_model('rst/pretrained/', 999)
    Trainer1.train(1000)
## Citation
