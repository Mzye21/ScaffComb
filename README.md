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
    > from GSG import AttnDGenerator
    > dt_L1000 = pickle.load(open('data/data_GSG_input_train_example.pkl','rb'))
    > dt_chembl = pickle.load(open('data/data_GSG_input_pretrain_example.pkl', 'rb'))
    > tokens = pickle.load(open('data/tokens.pkl','rb'))
    > # Pre-training with chembl scaffolds
    > Trainer0 = AttnDGenerator(tokens, dt_chembl, batch_size=128, lr = 0.0001,
                               device=device, path='rst/GSG_pretrained/')
    > Trainer0.train(1000)
    > # Train with L1000 scaffolds
    > Trainer1 = AttnDGenerator(tokens, dt_L1000, batch_size=128, lr = 0.0001,
                               device=device, path='rst/GSG_trained/')
    > Trainer1.load_model('rst/GSG_pretrained/', 999)
    > Trainer1.train(1000)
### Training DSP
    > from DSP import trainer
    > X, Y = pickle.load(open('data/data_DSP_input_example.pkl', 'rb'))
    > X_ = X.copy()
    > names = ['~'.join(sorted(x.split('~')[:2])) for x in X_.index.tolist()]
    > X_.index = names

    > X1 = X_.iloc[:, :1593]
    > X2 = X_.iloc[:, 1593:3186]
    > X3 = X_.iloc[:, 3186:]

    > model = AugSSP(1593, 978, device=device)
    > Trainer = trainer([X1, X2, X3, Y], model, batch_size=512, 
                        path = 'rst/DSP_trained/model')
    > Trainer.load_model()
    > Trainer.one_fold_train(500)
### Training SDSP
    > from SDSP import trainer
    > tokens = pickle.load(open('data/tokens.pkl','rb'))
    > dt = pickle.load(open('data/data_SDSP_input_example.pkl', 'rb'))
    > dt_train = [x[:40] for x in dt]
    > dt_test = [x[40:] for x in dt]
    > Trainer = trainer(tokens, dt_train, dt_test, latent_dim=512,
                        batch_size = 512, lr = 0.0001,
                        device=device, path='rst/SDSP_trained/model')
    > Trainer.train(100)

## Citation
