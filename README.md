# ST-fMRI
This repository contains PyTorch code for spatio-temporal deep learning on functional MRI data for phenotyping prediction. The original work was published at </url>[MLCN 2021](https://mlcnws.com/):


</url>[Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity](https://arxiv.org/abs/2109.03115)


# Downloading BHRC Dataset

Under request

# Installation

For PyTorch and dependencies installation, please follow instructions in [install.md](docs/install.md)

# Preprocessing 

## Gordon
In the folder /data/: 

### for 5 Fold CV


#### Psychiatric Diagnosis
'''
python preprocessing_nodetimeseries_trial.py --filename phenotypic.csv  --label dcany2010 --data_path /home/rc24/Documents/projects/INPD/GordonConnBOLD  --output_folder ../outputs/ --parcel Gordon
'''

#### Age regression
'''
python preprocessing_nodetimeseries_trial.py --filename phenotypic.csv  --label age_mri_baseline --data_path /home/rc24/Documents/projects/INPD/GordonConnBOLD  --output_folder ../outputs/ --regression True --parcel Gordon
'''
##### Age regression for normative model

'''
python preprocessing_nodetimeseries_normative.py --filename phenotypic.csv  --label age_mri_baseline --data_path /home/rc24/Documents/projects/INPD/GordonConnBOLD  --output_folder ../outputs/ --regression True --parcel Gordon
'''

## Schaefer

### for 5 Fold CV

#### Psychiatric Diagnosis
'''
python preprocessing_nodetimeseries_trial.py --filename phenotypic.csv  --label dcany2010 --data_path /home/rc24/Documents/projects/INPD/Schaefer_wave1  --output_folder ../outputs/ --parcel Schaefer
'''

#### for Longitudinal data (wave0 and wave1)
'''
python preprocessing_nodetimeseries_longitudinal.py \
    --train_val_filename phenotypic.csv \
    --test_filename phenotypic.csv \
    --train_val_path /home/rc24/Documents/projects/INPD/Schaefer_wave1 \
    --test_path /home/rc24/Documents/projects/INPD/Schaefer_wave2 \
    --label_train dcany2010 \
    --label_test dcFUPany \
    --output_folder ../outputs/ \
    --parcel Schaefer
'''

#### Age regression
'''
python preprocessing_nodetimeseries_trial.py --filename phenotypic.csv --label age_mri_baseline --data_path /home/rc24/Documents/projects/INPD/Schaefer_wave1 --output_folder ../outputs/ --parcel Schaefer --regression True
'''

#### Age regression for normative model

'''
python preprocessing_nodetimeseries_normative.py --filename phenotypic.csv  --label age_mri_baseline --data_path /home/rc24/Documents/projects/INPD/Schaefer_wave1  --output_folder ../outputs/ --regression True --parcel Schaefer
'''


# Training Brain-MS-G3D 

## Gordon 

### Train in whole data (all ROIs)
```
python ../tools/train_node_timeseries_networks.py --bs 32 --epochs 200 --gpu 0 --windows 80 --dropout 0.0 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx
```

### To training on each network
```
python ../tools/train_node_timeseries_networks.py --bs 32 --epochs 200 --gpu 1 --windows 40 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx --networks Default SMhand SMmouth Visual FrontoParietal Auditory None CinguloParietal RetrosplenialTemporal CinguloOperc VentralAttn Salience DorsalAttn
```

### To remove a network from training
```
python ../tools/train_node_timeseries_networks.py --bs 16 --epochs 300 --gpu 0 --windows 20 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx --networks Default SMhand SMmouth Visual FrontoParietal Auditory None CinguloParietal RetrosplenialTemporal CinguloOperc VentralAttn Salience DorsalAttn --remove_network True
```

### Ensemble network
'''
python ../tools/train_node_timeseries_networks.py --bs 32 --epochs 200 --gpu 0 --windows 30 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx --ensemble_networks Default SMhand
'''

For regression just add 
'''
--regression True
'''

## Schaefer Parcellation

### Train in whole data (all ROIs)
'''
python ../tools/train_node_timeseries_networks.py --bs 32 --epochs 300 --gpu 0 --windows 80 --dropout 0.0 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Schaefer2018_300Parcels_7Networks.xlsx
'''

### To training on each network
'''
python ../tools/train_node_timeseries_networks_copy.py  --bs 32 --epochs 300 --gpu 1 --windows 40 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Schaefer2018_300Parcels_7Networks.xlsx --networks Vis SomMot DorsAttn SalVentAttn Limbic Cont Default
'''

### Train normative model for age
'''
python ../tools/train_node_timeseries_normative.py --bs 8 --epochs 300 --gpu 0 --windows 42 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Schaefer2018_300Parcels_7Networks.xlsx  --dropout 0.7 --regression True
'''

### Longitudinal classification

'''
python ../tools/train_node_timeseries_longitudinal.py  --bs 32 --epochs 300 --gpu 1 --windows 40 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Schaefer2018_300Parcels_7Networks.xlsx --networks Vis SomMot DorsAttn SalVentAttn Limbic Cont Default
'''



# References 

This repository is based on the following repositories:


- repository: </url>[MS-G3D](https://github.com/kenziyuliu/MS-G3D) - paper: </url>[Z.Liu et al 2020](https://arxiv.org/abs/2003.14111)


- repository: </url>[ST-GCN-HCP](https://github.com/sgadgil6/cnslab_fmri) - paper: </url>[S.Gadgil et al 2020](https://arxiv.org/abs/2003.10613)

and 

- repository: </url>[ST-GCN](https://github.com/yysijie/st-gcn) - paper: </url>[S.Yan et al 2018](https://arxiv.org/abs/1801.07455)


# Citation

Please cite this work if you found it useful:

```
@misc{dahan2021improving,
      title={Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity}, 
      author={Simon Dahan and Logan Z. J. Williams and Daniel Rueckert and Emma C. Robinson},
      year={2021},
      eprint={2109.03115},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
      }
```

