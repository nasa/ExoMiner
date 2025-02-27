# ExoMiner

[//]: # (![ExoMiner Logo.]&#40;/others/images/exominer_logo.png&#41;)
<div style="text-align: center;">
    <img src="/others/images/exominer_logo.png" width="250" height="250" alt="Exominer Logo">
</div>

## Introduction

This project's mission is to develop, test, and deploy automated machine learning-based methods to sift ('mine') through 
transit photometry data from exoplanet survey missions such as Kepler and TESS and inform subject matter experts (SMEs) 
on potential transiting planet candidates.

### Current main goals

The main goals of the `ExoMiner` pipeline are:

1. Perform classification of transit signals in Kepler and TESS data;
2. Create vetted catalogs of Threshold Crossing Events (TCEs) for Kepler and TESS sector runs/cycles.
3. Validate new exoplanets in Kepler and TESS.

## Pipeline Overview

The pipeline consists of the following main blocks:

1. Data wrangling: perform operations on the data products that are used to generate the datasets to train and evaluate
   the models, and to run inference. This set of code involves tasks such as creating transit signal tables used to
   preprocess the data, perform analysis of the data, and evaluate results and rankings produced by the models. Code
   under `data_wrangling`.
2. Data preprocessing: preprocess data products (e.g. light curve FITS files)
   to generate a catalog of transit signal features to be used for training and evaluating models, and to run inference
   on it. Code under `src_preprocessing`.
3. Model training: train models on the preprocessed data.
    1. Hyper-parameter optimization: find a set of optimized hyper-parameters by running Bayesian optimization (vanilla
       or [BOHB](https://github.com/automl/HpBandSter)), or random search. Code under `src_hpo`.
4. Model evaluation: evaluate model performance against a holdout test set or by K-fold CV. Code under `src`
   and `src_cv`. Other additional experiments are shown below.
    1. Label noise injection: add artificial label noise to the training set and study the impact on model performance
       on a fixed test set. Code under `label_noise`.
    2. Adjusting size of training set: selectively sample the training set and study the impact on model performance on
       a fixed test set. Code under `size_training_set`.
    3. Transfer learning to different datasets: perform transfer learning experiments to apply models across different
       datasets (e.g. from Kepler to TESS). This includes analyzing impact of certain input features in the model
       performance, and fine-tuning certain layers.
    4. Explainability: design explainability studies that (1) help in improving and finding blindspots of the model,
       and (2) provide interpretation for the researcher/SME on the model output.
5. Inference: run trained models on a generated catalog of transit signals to generate a ranking that can be used for
   vetoing transit signals or exoplanet validation. Code under `src`.

## Data

All data used in this project are publicly available. Generally, the data used consist of:

- TCE and Objects of Interest (e.g. KOI and TOI catalogs) tables available in archives/respositories such as
  [NExSci](https://exoplanetarchive.ipac.caltech.edu/),
  [ExoFOP](https://exofop.ipac.caltech.edu/), [TEV](https://tev.mit.edu/data/) and
  [MAST](https://archive.stsci.edu/);
- Light curve and target pixel FITS files and other data products generated by the TESS Science Processing Operations 
- Center (SPOC) pipeline available in archives such as the [MAST](https://archive.stsci.edu/).

## Models

Models currently implemented in `models`:

1. `ExoMiner`: Multi-branch Convolutional Neural Network (CNN) whose design is inspired in the different diagnostic
   tests used in the Data Validation (DV) module of the SPOC pipeline.

![ExoMiner architecture.](others/images/exominer_architecture_tess.png)

2. Astronet: see https://github.com/google-research/exoplanet-ml and Shallue, Christopher J., and Andrew Vanderburg. "
   Identifying exoplanets with deep learning: A five-planet resonant chain around kepler-80 and an eighth planet around
   kepler-90." The Astronomical Journal 155.2 (2018): 94.
3. Exonet: see Ansdell, Megan, et al. "Scientific domain knowledge improves exoplanet transit classification with deep
   learning." The Astrophysical journal letters 869.1 (2018): L7.
4. Exonet-XS: see Ansdell, Megan, et al. "Scientific domain knowledge improves exoplanet transit classification with
   deep learning." The Astrophysical journal letters 869.1 (2018): L7. 5 MLP: Multi-layer Perceptron.

## References

For more detailed information see the following publications:
- ExoMiner 2021
  paper ["ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets"](https://arxiv.org/abs/2111.10009)
  , published 2022 February 17 in
  the [Astrophysical Journal, Volume 926, Number 2](https://iopscience.iop/articl/10.3847/1538-4357/ac4399/).
- ExoMiner w/ Multiplicity Boost, published 2023 June 26 in the [Astronomical Journal, Volume 166, Number 1](https://iopscience.iop.org/article/10.3847/1538-3881/acd344/)

## Credits

This work was developed by members of the Data Sciences Group, DASH, Intelligent Systems Division (Code-TI) at NASA Ames 
Research Center (NASA ARC).

- Main Contributors
  - Hamed Valizadegan<sup>1,2</sup>, hamed.valizadegan@nasa.gov
  - Miguel Martinho<sup>1,2</sup>, miguel.martinho@nasa.gov
  
- Collaborators
    - Doug Caldwell<sup>1,3</sup>
    - Jeff Smith<sup>1,3</sup>
    - Jon Jenkins<sup>1,3</sup>
    - Joseph Twicken<sup>1,3</sup>
    - Stephen Bryson<sup>1</sup>
  
- Active Developers
    - Adithya Giri<sup>7</sup> (Brown Dwarfs vs Planets Classification; Structured and Adversarial Training for Transit 
  Classification Robustness)
    - Josue Ochoa<sup>7</sup> (Transit Detection)

- Past developers 
    - Andrés Carranza <sup>2,5</sup> (Unfolded phase time series for transit signal classification)
    - Fellipe Marcellino<sup>2</sup> (Transit detection using Kepler data)
    - Jennifer Andersson<sup>4</sup> (Kepler to TESS transfer learning)
    - Kaylie Hausknecht<sup>2,6</sup> (Explainability framework using Kepler data)
    - Laurent Wilkens<sup>2</sup> (Kepler)
    - Martin Koeling<sup>4</sup> (Kepler to TESS transfer learning)
    - Nikash Walia<sup>2</sup> (Kepler)
    - Noa Lubin <sup>4</sup> (Kepler)
    - Pedro Gerum<sup>4</sup> (Kepler, Kepler non-TCE classification)
    - Patrick Maynard<sup>2,5,7</sup> (Kepler to TESS transfer learning)
    - Sam Donald<sup>4</sup> (Kepler to TESS transfer learning)
    - Theng Yang<sup>2,7</sup> (Label noise detection in Kepler data)
    - Hongbo Wei<sup>2,6,7</sup> (Kepler non-TCE classification, KOI classification, Kepler to TESS transfer learning)
    - Stuti Agarwal <sup>6</sup> (Difference image)
    - Joshua Belofsky <sup>2,5</sup> (Difference image)
    - Charles Yates <sup>2,5</sup> (Unfolded phase time series for transit signal classification, Kepler to TESS 
    transfer learning)
    - William Zhong <sup>5</sup> (Difference Image)
    - Ashley Raigosa<sup>7</sup> (TESS SPOC FFI)
    - Saiswaroop Thammineni<sup>7</sup> (Transit Encoding)
    - Kunal Malhotra<sup>7</sup> (Transit Detection)
    - Eric Liang<sup>7</sup> (Transit Encoding)
    - Ujjawal Prasad<sup>8</sup> (Transit Detection)

1 - NASA Ames Research Center (NASA ARC)\
2 - Universities Space Research Association (USRA)\
3 - The SETI Institute\
4 - NASA International Internship Program (NASA I<sup>2</sup>)\
5 - NASA Internships, Fellowships & Scholarships (NIFS)\
6 - Volunteer Internship Program (VIP)\
7 - NASA Office of STEM Engagement (OSTEM)\
8 - NASA-Chabot High School Learning Experience (NASA-CHSLE)

## Acknowledgements

We would like to acknowledge people that in some way supported our efforts:

- David Armstrong for an insightful discussion that improved our work.
- Megan Ansdell for providing information on their code and work.
- Resources supporting this work were provided by the NASA High-End Computing (HEC) Program through the NASA Advanced 
Supercomputing (NAS) Division at Ames Research Center.

[//]: # (- This work made use of the [gaia-kepler.fun]&#40;https://gaia-kepler.fun&#41; crossmatch database created by Megan Bedell.)
## Contacts
- Hamed Valizadegan (PI, USRA contractor),  hamed.valizadegan@nasa.gov
- Miguel Martinho (Sub-I, USRA contractor), miguel.martinho@nasa.gov
- Nikunj Oza (Group Lead, NASA ARC civil servant), nikunj.c.oza@nasa.gov

## Release Notes

First release of ExoMiner (v1.0). Expected features to be added in subsequent releases:
- TBD.

See the [NASA Open Source Agreement (NOSA)](others/licenses/nasa_open_source_agreement_ExoMiner-18792-1.pdf) for this software release.

For external collaborators, see the 
[Individual](ExoMiner_ARC-18792-1_Individual%20CLA.pdf) and [Corporate](others/licenses/Corporate_CLA_ExoMiner_ARC-18792-1.pdf) 
Contributor License Agreements.

* * * * * * * * * * * * * * 
Notices:

Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT. 

* * * * * * * * * * * * * * 