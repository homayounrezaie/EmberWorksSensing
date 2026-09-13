# EmberWorks — Wildfire Sensing Research Library

## Why thermography

Smoke is nearly transparent at long-wave infrared wavelengths. That single fact is
why this field exists: when a visible-light camera sees only a grey column, a
thermal sensor still sees the active fire edge underneath it. Everything else —
fire perimeters, spread rate, fire radiative power, intensity — follows from being
able to see the fire at all.

The industry has settled on this approach. OroraTech flies a CubeSat constellation
carrying both mid-wave and long-wave infrared bands, because MWIR picks up
high-intensity flaming while LWIR picks up cooler smoldering. Muon Space runs an
IR Data Products team for FireSat. Pano AI builds detection on camera networks.
Read the job ads these companies post and the same skills keep appearing:
experience with thermal and infrared imagery, radiometric calibration, sensor
characterization and the imaging chain, and machine learning applied to imagery.
Not fire science — sensor engineering.

## How it works

A thermal sensor measures radiance, not temperature. Radiance depends on the
surface's emissivity as well as its temperature, so converting a pixel to a
brightness temperature always assumes an emissivity you do not actually know.
Anything that raises radiance without a fire present produces a false detection,
and solar-heated bare rock raises it a great deal. This is the central problem:
on a sun-facing slope in the afternoon, rock and a small active fire can produce
the same LWIR signal.

The standard fix is a second band. Because the Planck function responds far more
sharply to temperature at 4 µm than at 11 µm, a hot sub-pixel fire brightens the
mid-wave band much more than the long-wave one, while warm rock brightens both
about equally — so the difference between the two separates them. This is what
MODIS and VIIRS do, and it is why operational sensors carry both. A single-band
LWIR instrument does not have that discriminator and has to substitute something
else for it. The papers here cover that problem from both ends: the sensor physics
that creates it, and the wildfire applications built on top of it.

## Contents

| Folder | Holds |
|---|---|
| [`thermography-general/`](thermography-general) | IR sensor physics — band selection, scene contrast, thermal image processing |
| [`thermography-wildfire/`](thermography-wildfire) | Thermal remote sensing applied to fire: detection, perimeters, radiative power, spread rate |
| [`fire-spread-modeling/`](fire-spread-modeling) | Fire behavior models, data assimilation, learned spread prediction, datasets |
| [`3d-reconstruction/`](3d-reconstruction) | Structure-from-motion, SLAM and Gaussian splatting, including thermal 3D |
| [`false-positives/`](false-positives) | Why non-fire surfaces read as fire in single-band LWIR (EN and Farsi) |

Also [`wildfire-companies.md`](wildfire-companies.md), a survey of companies working
in wildfire geospatial and remote sensing.

Papers in the two thermography folders are named `Author-Year-Short-Title.pdf`.
The other folders are named by method, which is how those works are usually cited.
