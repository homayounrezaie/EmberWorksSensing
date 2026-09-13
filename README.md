# EmberWorks — Wildfire Sensing Research Library

Reference collection supporting EmberWorks' work on airborne infrared wildfire
sensing: the sensor physics, the wildfire application literature, fire behavior
and spread modeling, and the 3D reconstruction methods used to turn imagery into
geometry.

## Layout

| Folder | Holds | Papers |
|---|---|---|
| [`thermography-general/`](thermography-general) | IR sensor physics — band selection, scene contrast, thermal image processing. Not wildfire-specific. | 4 |
| [`thermography-wildfire/`](thermography-wildfire) | Thermal remote sensing applied to wildfire: detection, perimeters, fire radiative power, spread rate. | 14 |
| [`fire-spread-modeling/`](fire-spread-modeling) | Fire behavior and spread models, data assimilation, learned spread prediction, datasets. | 25 |
| [`3d-reconstruction/`](3d-reconstruction) | Structure-from-motion, SLAM and Gaussian splatting, including thermal 3D reconstruction. | 9 |
| [`false-positives/`](false-positives) | Why non-fire surfaces read as fire in single-band LWIR. | 2 |

Also at the root: [`wildfire-companies.md`](wildfire-companies.md), a survey of
companies working in wildfire geospatial and remote sensing, and the EmberWorks
deck (`EmberWorks.pptx`) and cover image (`EmberWorks.png`).

**Filename convention:** `Author-Year-Short-Title.pdf` in the two thermography
folders. `fire-spread-modeling/` and `3d-reconstruction/` are named by method or
model, which is how these papers are usually referred to.

---

## thermography-general

IR sensor physics. The MWIR/LWIR band-selection papers matter because the
discriminator that satellite algorithms rely on — a mid-wave band alongside a
long-wave one — is unavailable on a single-band LWIR sensor.

| File | Paper | Authors | Year |
|---|---|---|---|
| `Stewart-2015-MWIR-vs-LWIR-Band-Merits.pdf` | Comparison of the relative merits of the midwave and longwave infrared bands for various target types using detected thermal contrast | Stewart | 2015 |
| `Jordan-2023-Scene-Contrast-Temperature-MWIR-LWIR.pdf` | Comparison of scene contrast temperature in mid-wave infrared and long-wave infrared | Jordan, Driggers, Furxhi et al. | 2023 |
| `Moore-2025-Scene-Contrast-Temperature-Two-Climates.pdf` | Scene contrast temperature in two different climates | Moore, Hendrick, Jacobs | 2025 |
| `Trongtirakul-2025-Entropy-Thermal-Image-Thresholding.pdf` | A Novel Entropy-Based Approach for Thermal Image Segmentation Using Multilevel Thresholding | Trongtirakul, Panetta, Grigoryan | 2025 |

## thermography-wildfire

Thermal remote sensing of active fire. Several of these are the standard
references for extracting fire intensity, perimeter and spread rate from
airborne IR.

| File | Paper | Authors | Year |
|---|---|---|---|
| `Wooster-2013-Thermal-RS-Active-Vegetation-Fires.pdf` | Thermal Remote Sensing of Active Vegetation Fires and Biomass Burning Events | Wooster, Roberts, Smith et al. | 2013 |
| `Paugam-2013-Handheld-Thermal-Imager-FRP-and-ROS.pdf` | Use of Handheld Thermal Imager Data for Airborne Mapping of Fire Radiative Power and Energy and Flame Front Rate of Spread | Paugam, Wooster, Roberts | 2013 |
| `Stow-2014-Fire-Spread-Rates-Repeat-Pass-ATIR.pdf` | Measuring fire spread rates from repeat pass airborne thermal infrared imagery | Stow, Riggan, Storey, Coulter | 2014 |
| `Allison-2016-Airborne-Optical-Thermal-RS-Review.pdf` | Airborne Optical and Thermal Remote Sensing for Wildfire Detection and Monitoring | Allison, Johnston, Craig, Jennings | 2016 |
| `Johnston-2017-Byram-Fire-Intensity-from-IR.pdf` | Direct estimation of Byram's fire intensity from infrared remote sensing imagery | Johnston, Wooster, Paugam et al. | 2017 |
| `Valero-2018-Automated-Fire-Perimeter-Edge-Detection.pdf` | Automated location of active fire perimeters in aerial infrared imaging using unsupervised edge detectors | Valero, Rios, Pastor, Planas | 2018 |
| `Sousa-2020-TIR-Near-Real-Time-Fire-Detection.pdf` | Thermal Infrared Sensing for Near Real-Time Data-Driven Fire Detection and Monitoring Systems | Sousa, Moutinho, Almeida | 2020 |
| `Paugam-2021-Orthorectification-Helicopter-IR-Imagery.pdf` | Orthorectification of Helicopter-Borne High Resolution Experimental Burn Observation from Infra Red Handheld Imagers | Paugam, Wooster, Mell et al. | 2021 |
| `Shennan-2023-Geovisualization-Landscape-Fire-Behavior.pdf` | Geovisualization and Analysis of Landscape-Level Wildfire Behavior Using Repeat Pass Airborne Thermal Infrared Imagery | Shennan, Stow, Nara, Schag, Riggan | 2023 |
| `Thornberry-2023-Lightweight-RS-Payload-FRP.pdf` | A Lightweight Remote Sensing Payload for Wildfire Detection and Fire Radiative Power Measurements | Thornberry, Gao, Ciciora et al. | 2023 |
| `Boroujeni-2024-AI-Enabled-UAS-Wildfire-Survey.pdf` | A comprehensive survey of research towards AI-enabled unmanned aerial systems in pre-, active-, and post-wildfire management | Boroujeni, Razi, Khoshdel et al. | 2024 |
| `Chen-2024-Remote-Sensing-Wildfire-Monitoring-Primer.pdf` | Remote sensing for wildfire monitoring: Insights into burned area, emissions, and fire dynamics | Chen, Morton, Randerson | 2024 |
| `Giesige-2025-Semi-Automated-Fire-Edge-Extraction.pdf` | Semi-Automated Extraction of Active Fire Edges from Tactical Infrared Observations of Wildfires | Giesige, Goldbeck-Dimon, Klofas, Valero | 2025 |
| `Guiop-Servan-2025-Remote-Sensing-Wildfire-Mapping-Review.pdf` | Remote Sensing for Wildfire Mapping: A Comprehensive Review of Advances, Platforms, and Algorithms | Guiop-Servan, Cotrina-Sanchez, Puerta-Culqui et al. | 2025 |

## fire-spread-modeling

### Fire behavior fundamentals

| File | Paper | Authors | Year |
|---|---|---|---|
| `Fuel Models.pdf` | Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model (RMRS-GTR-153) | Scott, Burgan | 2005 |
| `Rothermel Model.pdf` | The Rothermel Surface Fire Spread Model and Associated Developments: A Comprehensive Explanation | Andrews | 2018 |
| `wildfire simulation tools.pdf` | Comparative analysis of wildfire simulation tools: Discrepancies in Rothermel model-based software under varying wind and slope conditions | — | 2025 |

### Data assimilation

| File | Paper | Authors | Year |
|---|---|---|---|
| `Data Assimilation.pdf` | Data Assimilation for Wildland Fires: Ensemble Kalman Filters in Coupled Atmosphere-Surface Models | Mandel, Beezley, Coen, Kim | 2009 |
| `Fast Fourier Transform- EnKF.pdf` | Fast Fourier Transform Ensemble Kalman Filter with Application to a Coupled Atmosphere-Wildland Fire Model | Mandel, Beezley, Kondratenko | 2010 |
| `EnKF-FARSITE.pdf` | Wildfire Spread Prediction and Assimilation for FARSITE using Ensemble Kalman Filtering | Srivas, Artés, de Callafon, Altintas | 2016 |
| `Level-Set.pdf` | Towards Data Assimilation in Level-Set Wildfire Models Using Bayesian Filtering | Dabrowski, Huston, Hilton et al. | 2022 |
| `Latent-EnSF.pdf` | Latent-EnSF: A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Observation Data | Si, Chen | 2024 |
| `Fire-EnSF.pdf` | Fire-EnSF: Wildfire Spread Data Assimilation using Ensemble Score Filter | Shi, Wang, Liu | 2025 |
| `Ensemble Score Filter.pdf` | Robustness of the Ensemble Score Filter to the Type of Assimilated Observation Networks | Xiong, Liang, Bao, Zhang, Chipilski | 2025 |
| `LD-EnSF.pdf` | LD-EnSF: Synergizing Latent Dynamics with Ensemble Score Filters for Fast Data Assimilation with Sparse Observations | Xiao, Si, Chen | 2026 |
| `LEVDA.pdf` | LEVDA: Latent Ensemble Variational Data Assimilation via Differentiable Dynamics | Si, Chen | 2026 |
| `Data-Driven.pdf` | A Score Filter Enhanced Data Assimilation Framework for Data-Driven Dynamical Systems | Tang, Bausback, Bao, Zhang, Huynh | 2026 |
| `Partial Observability.pdf` | Robust Wildfire Forecasting under Partial Observability: From Reconstruction to Prediction | Yang, Zafari, Duan, Swindlehurst | 2026 |

### Learned spread prediction

| File | Paper | Authors | Year |
|---|---|---|---|
| `Fire Front Spread.pdf` | An Emulation Framework for Fire Front Spread | Bolt, Dabrowski, Huston, Kuhnert | 2022 |
| `Detecting Wildfires.pdf` | Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models | Pesonen, Hakala, Karjalainen, Koivumäki | 2024 |
| `Deep Learning.pdf` | Deep Learning for Wildfire Risk Prediction: Integrating Remote Sensing and Environmental Data | Xu, Li, Cheng et al. | 2025 |
| `DL-physics-based.pdf` | Integrating deep learning with physics-based model for predicting grassfire spread | Wadhwani, Zhang, Li, Sutherland, Moinuddin | 2025 |
| `Generative AI.pdf` | Generative AI for Predicting 2D and 3D Wildfire Spread: Beyond Physics-Based Models and Traditional Deep Learning | Xu, Zlatanova, Liang, Canbulat | 2025 |
| `Denoising Diffusion.pdf` | A probabilistic approach to wildfire spread prediction using a denoising diffusion surrogate model | Yu, Ghosh, Finn, Arcucci, Bocquet, Cheng | 2026 |
| `FireCast-fusion.pdf` | FireCast-fusion: Physics-Guided fusion of UAV RGB-thermal imagery and environmental data for near-term wildfire spread prediction | Abbas, Alshaikh, Obidallah et al. | 2026 |
| `Uncertainty-Aware.pdf` | Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting | Kondylatos, Papadopoulos, Camps-Valls, Papoutsis | 2026 |

### Datasets and benchmarks

| File | Paper | Authors | Year |
|---|---|---|---|
| `Next Day Wildfire Spread.pdf` | Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data | Huot, Hu, Goyal et al. | 2022 |
| `WildfireSpreadTS.pdf` | WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction | Gerard, Zhao, Sullivan | 2023 |
| `FLAME 3.pdf` | FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management | Hopkins, O'Neill, Marinaccio et al. | 2026 |

## 3d-reconstruction

Feed-forward SfM, SLAM and Gaussian splatting. The four thermal entries carry
these methods over to IR, where low texture and a narrow radiometric range make
correspondence harder than in RGB.

| File | Paper | Authors | Year |
|---|---|---|---|
| `MASt3R.pdf` | Grounding Image Matching in 3D with MASt3R | Leroy, Cabon, Revaud | 2024 |
| `MASt3R-SfM.pdf` | MASt3R-SfM: A Fully-Integrated Solution for Unconstrained Structure-from-Motion | Duisterhof, Zust, Weinzaepfel et al. | 2024 |
| `SplaTAM.pdf` | SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM | Keetha, Karhade, Jatavallabhula et al. | 2024 |
| `Thermal3D-GS.pdf` | Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis | Chen, Shu, Bai | 2024 |
| `ThermalGS.pdf` | ThermalGS: Dynamic 3D Thermal Reconstruction with Gaussian Splatting | Liu, Chen, Yan et al. | 2025 |
| `NTR-Gaussian.pdf` | NTR-Gaussian: Nighttime Dynamic Thermal Reconstruction with 4D Gaussian Splatting Based on Thermodynamics | Yang, Liu, Cui et al. | 2025 |
| `Point3R.pdf` | Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory | Wu, Zheng, Zhou, Lu | 2025 |
| `PLANING.pdf` | PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction | Jiang, Ren, Li et al. | 2026 |
| `Thermal4D.pdf` | Thermal4D: Physics-Driven Gaussian Splatting for Dynamic Thermal Scene Reconstruction | Zhong, Xu | 2026 |

## false-positives

Technical note, in English and Farsi: *Why Non-Fire Surfaces Are Detected as Fire
in Airborne LWIR Imagery*. Using the radiative transfer equation, the surface
energy balance, the Dozier sub-pixel mixing model and laboratory spectral
emissivity data, it works through why solar-heated bare rock on sun-facing slopes
emits LWIR radiance that cannot be separated from a small active fire, and what
that means for designing a detection algorithm on a single-band sensor.

| File | Language |
|---|---|
| `LWIR_false_fire_EN.pdf` | English |
| `LWIR_false_fire_FA.pdf` | Farsi |
