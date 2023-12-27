# Fall Detection

Empirical study of TSC SOTA models on the FallAllD dataset

## Setup

For the most convenience, the code can be run using Google Colab by downloading the repository to Google Drive and working from there. In case of using a local environment, the libraries in the [requirements.txt](https://github.com/almasrifi-rami/fall_detection/blob/main/requirements.txt) need to be installed.

Also, the data need to be downloaded from [IEEDataPort](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living), then processed using [FallAllD_to_PYTHON_Structure.py](https://github.com/almasrifi-rami/fall_detection/blob/main/src/utils/FallAllD_to_PYTHON_Structure.py) to create a pickle file which can be used for loading the data and training the model.

## Data Loading

After processing the data files using [FallAllD_to_PYTHON_Structure.py](https://github.com/almasrifi-rami/fall_detection/blob/main/src/utils/FallAllD_to_PYTHON_Structure.py), the data can be loaded using two functions from [utils.utils.py](https://github.com/almasrifi-rami/fall_detection/blob/main/src/utils/utils.py) module:

1. [load_df()](https://github.com/almasrifi-rami/fall_detection/blob/1e78539ef9c9c4ccbd2705dc7fa16851039e0f57/src/utils/utils.py#L25) for loading the data into a pandas DataFrame for data exploration and analysis
2. [load_data()](https://github.com/almasrifi-rami/fall_detection/blob/1e78539ef9c9c4ccbd2705dc7fa16851039e0f57/src/utils/utils.py#L194) for loading the data into a format suitable for most machine learning models using tensorflow (instances, timesteps, variables)

## Models

6 models were used in total and the code are provided in [utils.models.py](https://github.com/almasrifi-rami/fall_detection/blob/main/src/utils/models.py). The models used are FCN, ResNet, InceptionTime, LSTM-FCN, ROCKET, and MINIROCKET, all of which achieve state-of-the-art (SOTA) performance on time series classification (TSC) tasks.

## Results

An accuracy of approx. 95% can be achieved by using MINIROCKET to transform the data then fitting a linear classifier such as ridge regression classifier. And while the results show MINIROCKET achieves higher accuracy than the other models, there is no significant difference in the accuracy of MINIROCKET in comparison to the other models.

## References

### Data

@data{bnya-mn34-20,
doi = {10.21227/bnya-mn34},
url = {https://dx.doi.org/10.21227/bnya-mn34},
author = {SALEH, Majd and LE BOUQUIN JEANNES, Régine},
publisher = {IEEE Dataport},
title = {FallAllD: A Comprehensive Dataset of Human Falls and Activities of Daily Living},
year = {2020} }

### Models

1. FCN + ResNet + InceptionTime
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2019},
  volume                   = {33},
  number                   = {4},
  pages                    = {917--963},
}
2. LSTM-FCN
@misc{Karim2018,
  Author = {Fazle Karim and Somshubra Majumdar and Houshang Darabi and Samuel Harford},
  Title = {Multivariate LSTM-FCNs for Time Series Classification},
  Year = {2018},
  Eprint = {arXiv:1801.04503},
}
3. ROCKET
@article{dempster_etal_2020,
  author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
  title   = {{ROCKET}: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels},
  journal = {Data Mining and Knowledge Discovery},
  year    = {2020},
  volume  = {34},
  number  = {5},
  pages   = {1454--1495}
}
4. MINIROCKET
@inproceedings{dempster_etal_2021,
  author    = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title     = {{MiniRocket}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  address   = {New York},
  year      = {2021},
  pages     = {248--257}
}
