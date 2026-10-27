# EmberWorks — Wildfire  Research

## Contents

| Folder | Holds |
|---|---|
| [`thermography-general/`](thermography-general) | IR sensor physics — band selection, scene contrast, thermal image processing |
| [`thermography-wildfire/`](thermography-wildfire) | Thermal remote sensing applied to fire: detection, perimeters, radiative power, spread rate |
| [`fire-spread-modeling/`](fire-spread-modeling) | Fire behavior models (Rothermel, FARSITE, FSim, WRF-SFIRE), data assimilation, learned and physics-informed spread prediction, datasets |
| [`canadian-fire-systems/`](canadian-fire-systems) | Canadian Forest Fire Danger Rating System: FBP System and FWI System |
| [`crown-fire-and-spotting/`](crown-fire-and-spotting) | Where surface spread models break down: crown fire and spotting |
| [`model-inputs/`](model-inputs) | Fuel (LANDFIRE), weather (HRRR) and wind (WindNinja, aircraft-derived wind) inputs, plus notes |
| [`model-validation/`](model-validation) | Evaluating fire models against observed fires |
| [`suppression/`](suppression) | Suppression resource allocation and measuring suppression effectiveness |
| [`3d-reconstruction/`](3d-reconstruction) | Structure-from-motion, SLAM and Gaussian splatting, including thermal 3D |
| [`false-positives/`](false-positives) | Why non-fire surfaces read as fire in single-band LWIR (EN and Farsi) |

Also [`wildfire-companies.md`](wildfire-companies.md), a survey of companies working
in wildfire geospatial and remote sensing, and [`fire-researchers.txt`](fire-researchers.txt).

## Not yet in the library

Paywalled or not downloadable by script. Download manually and drop into the folder shown.

| Paper | Folder | Link |
|---|---|---|
| Lautenberger 2013, *Wildland fire modeling with an Eulerian level set method and automated calibration* (ELMFIRE), Fire Safety Journal 62. The ELMFIRE 1.1 User Guide in `fire-spread-modeling/` covers the same formulation | `fire-spread-modeling/` | [doi:10.1016/j.firesaf.2013.08.014](https://doi.org/10.1016/j.firesaf.2013.08.014) |
| Tymstra et al. 2010, *Development and structure of Prometheus*, NOR-X-417 (free) | `canadian-fire-systems/` | [publications.gc.ca](https://publications.gc.ca/site/eng/9.568367/publication.html) |
| Van Wagner 1977, *Conditions for the start and spread of crown fire*, Can. J. For. Res. 7. Its criteria are restated in Scott & Reinhardt 2001, already in `crown-fire-and-spotting/` | `crown-fire-and-spotting/` | [doi:10.1139/x77-004](https://doi.org/10.1139/x77-004) |
| Cruz & Alexander 2010, *Assessing crown fire potential in coniferous forests of western North America*, IJWF 19 | `crown-fire-and-spotting/` | [doi:10.1071/WF08132](https://doi.org/10.1071/WF08132) |
| Wooster et al. 2005, *Retrieval of biomass combustion rates and totals from fire radiative power observations*, JGR 110 | `thermography-wildfire/` | [doi:10.1029/2005JD006318](https://doi.org/10.1029/2005JD006318) |
| Dowell et al. 2022, *The High-Resolution Rapid Refresh (HRRR), Part I* | `model-inputs/` | [doi:10.1175/WAF-D-21-0151.1](https://doi.org/10.1175/WAF-D-21-0151.1) |
