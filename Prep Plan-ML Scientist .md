# Prep Plan — EmberWorks ML Wildfire Scientist

https://workforcenow.adp.com/mascsr/default/mdf/recruitment/recruitment.html?cid=92144dc3-94dc-44b9-8a18-90099bf3a77c&ccId=19000101_000001&type=JS&lang=en_US&jobId=561653

## Step 1: Read all the papers

Read every paper in your [EmberWorksSensing repo](https://github.com/homayounrezaie/EmberWorksSensing), group by group in the order of Table 1. For each paper, write one line in your notes: the method, the data, and the main result.

- [ ] Fire behaviour equations
- [ ] Crown fire and spotting
- [ ] Spread simulators
- [ ] Data assimilation
- [ ] Physics-informed ML
- [ ] ML and deep learning spread
- [ ] Fuel and weather inputs
- [ ] Model validation
- [ ] Fire behaviour from thermal data
- [ ] Suppression
- [ ] Background groups (thermal sensing, detection, reviews, risk, 3D)

## Step 2: Table 1 — papers by method

| Method group | Papers in the repo |
| --- | --- |
| 1. Fire behaviour equations | Rothermel review (Andrews 2018), Scott & Burgan fuel models, FBP System (ST-X-3, 1992), FWI System (Van Wagner 1987) |
| 2. Crown fire and spotting | Rothermel 1991 crown fire, Scott & Reinhardt 2001 crown fire potential, Albini 1979 spotting |
| 3. Spread simulators | FARSITE (Finney 1998), ELMFIRE 1.1 user guide (Lautenberger 2026), WRF-SFIRE (Mandel 2011), FSim (Finney 2011), wildfire simulation tools review (2025) |
| 4. Data assimilation | Mandel 2009 EnKF, EnKF-FARSITE (Cocos Fire, 2016), Level-Set Bayesian filtering (2022), FFT-EnKF (2018), Fire-EnSF (2025), Ensemble Score Filter, Latent-EnSF, LD-EnSF, LEVDA, score filter for data-driven systems (2026) |
| 5. Physics-informed ML | Bayesian PINN for wildfire data assimilation (Dabrowski 2023), deep learning + physics-based model (2025), fire front emulation (Bolt 2022) |
| 6. ML and deep learning spread | Next Day Wildfire Spread, WildfireSpreadTS, denoising diffusion surrogate, Generative AI 2D/3D spread, FireCast-fusion (2026), partial observability (2026) |
| 7. Fuel and weather inputs | LANDFIRE fuel data (Reeves 2009), WindNinja (Forthofer 2014), aircraft meteo-particle wind (Sun 2018) |
| 8. Model validation | Filippi 2014, evaluation of fire models on 80 observed fires |
| 9. Fire behaviour from thermal data | Stow 2014 repeat-pass spread rates, Shennan 2023, Paugam 2013 FRP and spread rate, Johnston 2017 Byram intensity, Valero 2018 and Giesige 2025 fire edges, Paugam 2021 orthorectification, Ma 2026 temperature error |
| 10. Suppression | Wheatley et al. (IR suppression effectiveness), Delazeri 2026 suppression optimization |
| Background: thermal sensing | Stewart 2015 MWIR vs LWIR, Jordan 2023 and Moore 2025 scene contrast, Trongtirakul 2025 thermal thresholding, your LWIR false-fire note |
| Background: detection and platforms | UAV real-time fire segmentation, FLAME 3, WIT-UAS, UAV fleet monitoring, Sousa 2020, Thornberry 2023 FRP payload, Boroujeni 2024 AI-UAS survey |
| Background: reviews | Allison 2016, Chen 2024 primer, Guiop-Servan 2025, Wooster 2013 chapter |
| Background: risk and danger | Deep learning wildfire risk (Xu), uncertainty-aware danger forecasting (Kondylatos) |
| Background: 3D reconstruction | MASt3R, MASt3R-SfM, Point3R, SplaTAM, PLANING, ThermalGS, Thermal3D-GS, Thermal4D, NTR-Gaussian |

## Step 3: Table 2 — papers to reproduce

One paper per method group, chosen for open data or code and closeness to ORBIT's problem. Match the paper's own numbers on the paper's own data first; that proves your version is correct.

| Method group | Reproduce | What to match | Data or code |
| --- | --- | --- | --- |
| 1. Fire behaviour equations | Rothermel review (Andrews 2018) | Its worked rate-of-spread examples | Equations and examples in the paper |
| 2. Crown fire and spotting | Scott & Reinhardt 2001 | Torching and crowning indices for its examples | Equations in the report |
| 3. Spread simulators | ELMFIRE 1.1 user guide | Its verification cases | ELMFIRE on GitHub (lautenberger/elmfire) |
| 4. Data assimilation | EnKF-FARSITE, Cocos Fire (Srivas 2016) | Growing error without updates, stable error with hourly EnKF | Synthetic "true" fire; your own level-set code |
| 5. Physics-informed ML | Dabrowski 2023 Bayesian PINN | Spread recovered with uncertainty | Code on GitHub (jjdabr/BPINN-Wildfire) |
| 6. ML and deep learning spread | Next Day Wildfire Spread | Its reported baseline score | Public dataset; WildfireSpreadTS as a stretch |
| 7. Fuel and weather inputs | Sun 2018 meteo-particle model | A wind field rebuilt from aircraft data | Code on GitHub (junzis/meteo-particle-model) |
| 8. Model validation | Filippi 2014 | Its scoring methods (Sørensen, arrival time) | The paper says its data and code are public |
| 9. Fire behaviour from thermal data | Stow 2014 | Spread rate between two thermal passes | Any open repeat-pass thermal dataset |
| 10. Suppression | — | Read only | — |

Before each reproduction, spend 15 minutes confirming the code or data is available; if not, switch to the next paper in the same group.

## Step 4: Apply all the methods to one fire

A comparison is only fair when every method gets the same inputs, the same observations and the same scoring.

- **Benchmark dataset:** the [NIROPS daily fire perimeter progression dataset](https://data.mendeley.com/datasets/95rj5d379g/1) (Western US, 2020–2024): 12,705 perimeters from 737 wildfires, digitised from airborne infrared imagery flown daily or more often. It is the closest open match to ORBIT's data. Licence CC BY 4.0.
- **Benchmark fire:** the 2021 Dixie Fire in California (`CA-BTU-009205_Dixie`). It has 75 NIROPS perimeters from 15 July to 13 October 2021, daily without a gap until 13 September, and it is in your own detection project. Only the perimeters are public: the original infrared imagery is not, and the old NIFC FTP folders for 2021 no longer exist. Check that 2021 is inside WildfireSpreadTS coverage so the ML model can run on it too.
- **Hold it out:** remove the Dixie Fire from any ML training data.
- **Same inputs for all:** LANDFIRE fuels and terrain, HRRR weather, the same starting NIROPS perimeter and start time.
- **Same forecast windows:** score each method against the next NIROPS perimeter (about 24 hours), plus 6 and 12 hours where VIIRS detections allow.
- **Methods to run:** persistence (fire stays where it is), Rothermel-based ELMFIRE, ELMFIRE with EnKF updates, the physics-informed network, the deep learning spread model, and a hybrid where ML corrects ELMFIRE.
- **Synthetic fire:** use a simulated fire with a known true answer for data assimilation and the physics-informed network, since a real fire has no perfect truth.
- **Supporting datasets:** [RxCADRE 2011–2012](https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0007) (calibrated airborne LWIR every 3–4 seconds over prescribed burns) to check fire behaviour code against measured spread rate and FRP; the [Canadian Fire Spread Dataset](https://www.nature.com/articles/s41597-024-03436-4) (3,269 fires, 2002–2021, 180 m daily) as an optional test on Canadian fires.

## Step 5: Compare the methods

Score every method on the benchmark fire with the same criteria, so you can say which method fits ORBIT's data and why, with numbers. Speed and memory matter, but for EmberWorks the deciding questions are whether a method can take in a new pass every orbit and how long its forecast stays useful.

| Criterion | What to measure | Why it matters for ORBIT |
| --- | --- | --- |
| Accuracy | Sørensen or Jaccard overlap, arrival-time error, rate-of-spread error | Is the predicted fire where the real one went? |
| Beats persistence | Same scores for "fire stays where it is" | A method that can't beat this isn't useful |
| Skill horizon | Hours until accuracy falls below persistence | How far ahead can crews trust it? |
| Uncertainty | Ensemble spread vs actual error; coverage of probability maps | Crews need to know how confident to be |
| Speed | Run time per forecast (and per ensemble member) on CPU and GPU | Must finish well inside one orbit (about 2 minutes) |
| Scaling | How run time grows with grid size and ensemble size (time complexity) | Large fires and big ensembles |
| Memory | Peak RAM or GPU memory per run | Cost and what hardware is needed |
| Data needs | Inputs required (fuel, weather, terrain) and training data size | Some inputs are missing or out of date in real fires |
| Takes new observations | Can it be updated from each pass, and how easily? | ORBIT's main advantage |
| Sensitivity to inputs | Change in result when wind or fuel is perturbed | Shows which input errors dominate |
| Source of errors | Model structure, inputs, starting perimeter, or observations | The "model or data?" question |
| Generalisation | Scores on held-out whole fires, new regions, extreme days | Rare extreme days matter most |
| Physical consistency | Burned area never shrinks; no spread into water or rock | Trust and safety |
| Explainability | Can you say why it predicted that, in physical terms? | Crews won't use a black box |
| Robustness to bad data | Effect of false detections (hot rock) and missed passes | Real thermal data is noisy |
| Maturity | Open source, documentation, operational use, licence | Effort and risk to adopt |

Keep one results table in your repo with a row per method and these columns. Fill it in as each method finishes; the finished table becomes the most persuasive slide in your interview deck.
