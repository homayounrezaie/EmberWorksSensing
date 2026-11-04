# Model inputs — fuel, weather, terrain

## LANDFIRE (fuel + terrain, US only)

- 30 m rasters for CONUS, Alaska, Hawaii. Background paper: Reeves et al. 2009 (in this folder).
- Surface fuel: **FBFM13** (Anderson 13) and **FBFM40** (Scott & Burgan 40, see
  `fire-spread-modeling/Fuel Models.pdf`).
- Canopy fuel, needed for crown fire: **CC** canopy cover, **CH** canopy height,
  **CBH** canopy base height, **CBD** canopy bulk density.
- Terrain: elevation, slope, aspect.
- These 8 layers stack into a FARSITE/FlamMap **landscape (.lcp)** file, the standard
  input for Rothermel-based simulators.
- Known weaknesses:
  - Fuels lag reality: recent burns, treatments and beetle kill appear only after a
    disturbance update.
  - CBH and CBD are modelled, not measured, so crown fire predictions inherit that error.
  - Fuel model assignment is a crosswalk from vegetation type, so local mismatches are common.
- No coverage in Canada. There the equivalent is the FBP System fuel types
  (`canadian-fire-systems/`).

## HRRR (weather, US)

- NOAA High-Resolution Rapid Refresh: 3 km grid, convection-allowing, built on WRF-ARW.
- New run every hour (hourly cycling). Forecasts go out 18 h; the 00/06/12/18Z runs go out 48 h.
- Assimilates radar reflectivity, so it handles convective outflows better than coarser models.
- HRRR-Smoke uses satellite FRP for smoke emissions, so the same FRP physics as our thermal work.
- 10 m winds are 3 km cell averages. They miss terrain channelling, so downscale with
  WindNinja (Forthofer et al. 2014, in this folder).
- Archive on AWS Open Data (`noaa-hrrr-bdp-pds`); the Python package `herbie` reads it.
- Reference: Dowell et al. 2022, *Weather and Forecasting* 37(8), 1371–1395,
  doi:10.1175/WAF-D-21-0151.1 (Part I, system description). Not in the library yet.
- In Canada the equivalent is ECCC's HRDPS (~2.5 km).

## Wind from our own aircraft

- Sun et al. 2018 (TU Delft, in this folder): reconstructs a wind field from aircraft
  surveillance data with a "meteo-particle" model.
- This is the basis for using air attack aircraft as wind sensors over the fire.
