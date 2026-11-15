"""Page content for the web demo (static HTML blocks + short labels).

Static blocks ({{...}} placeholders in web_template.html) are substituted at
build time; `labels` travel in the page's JSON meta for the JS.
"""
from __future__ import annotations

SOURCES = [
    dict(id="FLAME 3", what="nadir thermal plot subset, Hanna Hammock plot 1: stabilised, georeferenced radiometric "
                            "thermal frames (°C) every 5 s, GCPs, burn map and the crew's weather note",
         cite="Hopkins, B.; O'Neill, L.; Marinaccio, M.; Rowell, E.; Parsons, R.; Flanary, S.; Nazim, I.; Seielstad, C.; "
              "Afghah, F. 2024. FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire "
              "Management. arXiv:2412.02831. Data: https://www.kaggle.com/dsv/8724543 and https://dx.doi.org/10.21227/w0mz-aq48"),
    dict(id="KTLH METAR", what="wind direction and open 10 m wind (Tallahassee airport, 30 km S), 28 Feb 2023",
         cite="Iowa Environmental Mesonet ASOS/AWOS archive, station TLH. https://mesonet.agron.iastate.edu/request/download.phtml"),
    dict(id="USGS 3DEP", what="1 m lidar digital elevation model (FL Hurricane Michael 2020 project) -> slope and aspect",
         cite="U.S. Geological Survey 3D Elevation Program, 3DEPElevation ImageServer. https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"),
    dict(id="Sentinel-2 L2A", what="NIR / SWIR reflectance over the plot, every clear scene 2019 – 27 Feb 2023 (tile 16RGU) "
                                     "-> NDII and NDWI for live fuel moisture",
         cite="Copernicus Sentinel data, ESA; COGs via Element 84 Earth Search on AWS. https://earth-search.aws.element84.com/v1"),
    dict(id="Landsat 8/9 C2 L2", what="surface reflectance over the plot, 2019 – 27 Feb 2023 -> NDII cross-check",
         cite="U.S. Geological Survey Landsat Collection 2 Level-2; via Microsoft Planetary Computer. https://planetarycomputer.microsoft.com"),
    dict(id="Sentinel-1 RTC", what="VH / VV backscatter over the plot, 2019 – 27 Feb 2023 -> microwave cross-check",
         cite="Copernicus Sentinel-1 data, ESA; radiometrically terrain-corrected on Microsoft Planetary Computer."),
    dict(id="LANDFIRE 2022", what="Scott & Burgan fuel model (FBFM40), canopy cover, height, base height and bulk density, 30 m",
         cite="LANDFIRE 2022 (LF 2.3.0), U.S. Department of the Interior / U.S. Department of Agriculture. https://landfire.gov"),
]


def _steps(items):
    return "\n".join(f'<div class="step"><div class="n">{i}</div><h4>{h}</h4><p>{p}</p></div>'
                     for i, (h, p) in enumerate(items, 1))


SITE = dict(
    page="index.html", data="hanna_hammock.json",
    hide=["scores", "how"],  # per user request: no score tiles / per-snapshot chart / how-it-works / limitations box
    title="Head Fire Discovery",
    description="FLAME 3 drone radiometric LWIR (Hanna Hammock, 28 Feb 2023): FlamMap, FARSITE-style Rothermel, the "
                "Canadian FBP System and persistence forecasts vs. a spread equation discovered by sparse regression, "
                "scored on held-out 30 s snapshots.",
    h1="Fire-spread law from drone LWIR: data-discovered vs. operational models",
    subtitle="FLAME 3 nadir thermal plot 1, <b>Hanna Hammock</b> prescribed burn (Tall Timbers Research Station, Florida, "
             "28 Feb 2023): a radiometric thermal frame every 5 s for 25 min over a ~100 × 90 m pine stand, analysed on "
             "a 0.5 m grid. A crew ignition line along the plot's west edge at 15:58 launches a single head fire east "
             "across the plot. Each 30 s snapshot is forecast by FlamMap (minimum travel time), FARSITE-style Rothermel "
             "wavelets, the Canadian FBP System and persistence, and by a sparse-regression (STLSQ) equation fitted to "
             "the early snapshots; all get the same inputs (fuel moisture from the thermal image, lidar terrain, "
             "LANDFIRE fuels and canopy, airport wind) and are scored on later snapshots.",
    map_caption="Switch layers on and off in the Layers panel (left), as in a GIS. Fire: the white line is the observed "
                "burnt-area perimeter at snapshot <i>p</i>; each coloured line is a model's forecast of it from snapshot "
                "<i>p−1</i>, 30 s earlier (new ignitions seen in the imagery in between are given to every model). "
                "Base map: the drone thermal image (brightness temperature, emissivity 1). Burnt = ever hotter than 120 °C "
                "(or warm ash &gt; 60 °C in the first frame).",
    eqvars="θ angle between front normal and wind (from 240°) · FRFD radiant flux (heat flux) at the front, σT⁴ of the LWIR "
           "brightness temperature above background (kW/m²) · κ front curvature (1/m, &gt;0 bulging into fuel) · U midflame "
           "wind (m/s) · R_Roth, R_FBP the Rothermel and FBP rates at the point and direction, from the LANDFIRE fuel model, "
           "thermal-image dead and satellite live fuel moisture, canopy, slope and Buildup Index (m/s) · ROS normal spread rate (m/s).",
    panel_caption="Grey = burnt at the start snapshot; filled = new burnt area 30 s later (actual or forecast); "
                  "outline = actual next perimeter.",
    steps=_steps([
        ("Watch", "296 stabilised, GCP-georeferenced radiometric frames (°C) from a UAV hovering 122 m above the plot are "
                  "reprojected onto one north-up 0.5 m grid and sampled every 30 s."),
        ("Extract the front", "Burnt = ever hotter than 120 °C in any 5 s frame; flaming = hotter than 300 °C."),
        ("Offer candidate terms", "Eight terms from the wind angle θ, the radiant flux at the front and curvature."),
        ("Keep the few that matter", "STLSQ drops terms below λ (chosen by cross-validation on training data only)."),
    ]),
    limitations="",
    footer='Data: <a href="https://www.kaggle.com/datasets/brycehopkins/flame-3-nadir-thermal-plot-subset" target="_blank" '
           'rel="noopener">FLAME 3 drone radiometric thermal dataset</a>, Hanna Hammock plot 1 (Hopkins et al. 2024) · '
           '<a href="https://dx.doi.org/10.21227/w0mz-aq48" target="_blank" rel="noopener">IEEE DataPort</a> · '
           '<a href="https://arxiv.org/abs/2412.02831" target="_blank" rel="noopener">paper</a>',
    inputs_caption="Surface temperature and fuel moisture follow the snapshot on the map: at each snapshot, the mean brightness "
                   "temperature of the 30 s of frames up to it (first map) (unburnt fuel at least 2 m from burnt ground and below 45 °C; other cells take the estimate of "
                   "valid neighbours within a few metres) sets the humidity at the fuel surface, e<sub>air</sub> / "
                   "e<sub>sat</sub>(T<sub>fuel</sub>), with the crew's air temperature and humidity; the Simard (1968) "
                   "equilibrium moisture follows. It is an estimate, not a measurement — no fuel samples were taken, and "
                   "the warm north-east corner is probably old black ash from the earlier slow front, not dry fuel. Terrain "
                   "smoothed at 2 m. Midflame wind = 20-ft wind × wind adjustment factor from LANDFIRE canopy cover, height "
                   "and base height (sheltered, Albini &amp; Baughman 1979 / Andrews 2012) or the fuel-bed depth where the "
                   "canopy is sparse; the crew's in-stand reading was 0.45 m/s. Live fuel moisture is not visible in LWIR: "
                   "it comes from Sentinel-2 NDII placed within its 2019–2023 range at the plot (map at 20 m; Landsat and Sentinel-1 agree). Pale = outside the plot.",
    methods="Rothermel (1972), Albini (1976), Scott &amp; Burgan (2005) fuel models, Simard (1968) EMC, Albini &amp; "
            "Baughman (1979) and Andrews (2012) wind adjustment, Anderson (1983) length-to-breadth; FARSITE Huygens "
            "wavelets (Finney 1998, Richards 1990); FlamMap minimum travel time (Finney 2002, re-implemented — not the "
            "FlamMap software); Canadian FBP System fuel type C-7 with slope-equivalent wind (Forestry Canada 1992, "
            "ST-X-3) with the Fire Weather Index codes (Van Wagner 1987) from KTLH weather; live fuel moisture from Sentinel-2 "
            "NDII by the relative-greenness method (Burgan &amp; Hartford 1993); SINDy / STLSQ (Brunton et al. 2016) with physics sign priors and the threshold chosen by held-out "
            "forecast skill; paired Wilcoxon signed-rank test.",
    data_heading="Data (openly published by the FLAME 3 authors; licence not stated, so cite and check terms before "
                 "redistribution):",
    sources=SOURCES,
    labels=dict(frame="Snapshot", frames="snapshots", clock="EST", wind="midflame wind", wind_short="airport",
                I_desc="radiant flux (from brightness temperature)",
                pre_fire="Snapshot before any forecast.", first_fire="First snapshot — no forecast yet."),
    color=dict(unit="°C", lo=20.0, hi=600.0, ticks=[20, 50, 100, 200, 400, 600]),
)
