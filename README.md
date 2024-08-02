# ST-fMRI
This repository contains PyTorch code for spatio-temporal deep learning on functional MRI data for phenotyping prediction. The original work was published at </url>[MLCN 2021](https://mlcnws.com/):


</url>[Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity](https://arxiv.org/abs/2109.03115)


## Downloading HCP Dataset

HCP data can be directly downloaded from </url>[Human Connectome Project](https://db.humanconnectome.org/)

## Installation

For PyTorch and dependencies installation, please follow instructions in [install.md](docs/install.md)

## Preprocessing 

In the folder /data/: 

'''
python preprocessing_nodetimeseries_trial.py --filename phenotypic.csv  --label TOTAL_DAWBA --data_path /home/rc24/Documents/projects  --output_folder ../outputs/
'''

```
 python preprocessing_nodetimeseries.py --filename phenotypic.csv  --label gender --data_path /home/rc24/Documents/projects  --output_folder ../outputs/
```


## Training Brain-MS-G3D 

For DAWBA classification

```
python ../tools/train_node_timeseries.py --nodes 333 --bs 32 --epochs 200 --gpu 0 --windows 30 --dropout 0.3 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx
``

```
python ../tools/train_node_timeseries_networks.py --nodes 333 --bs 32 --epochs 200 --gpu 0 --windows 30 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx --networks Default SMhand SMmouth Visual FrontoParietal Auditory None CinguloParietal RetrosplenialTemporal CinguloOperc VentralAttn Salience DorsalAttn
```

'''
python ../tools/train_node_timeseries_networks.py --nodes 333 --bs 32 --epochs 200 --gpu 0 --windows 30 --data_path ../outputs --parcel_path /home/rc24/Documents/projects/INPD/Parcels.xlsx --ensemble_networks Visual CinguloOperc
'''

For fluid intelligence regression


```
python ./tools/train_node_timeseries.py --nodes 25 --bs 64 --epochs 100 --gpu 0 --windows 100 --fluid --data_path path_to_data
```

## Tensorboard

Starting tensorboard visualisation

```
tensorboard --logdir ./logs/MS-G3D/
```


## Docker support 

**Coming soon**

## References 

This repository is based on the following repositories:


- repository: </url>[MS-G3D](https://github.com/kenziyuliu/MS-G3D) - paper: </url>[Z.Liu et al 2020](https://arxiv.org/abs/2003.14111)


- repository: </url>[ST-GCN-HCP](https://github.com/sgadgil6/cnslab_fmri) - paper: </url>[S.Gadgil et al 2020](https://arxiv.org/abs/2003.10613)

and 

- repository: </url>[ST-GCN](https://github.com/yysijie/st-gcn) - paper: </url>[S.Yan et al 2018](https://arxiv.org/abs/1801.07455)


## Citation

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

