"""Pipeline configuration for the FLAME 3 Hanna Hammock case.  Everything
tunable lives here."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# FLAME 3 nadir thermal plot 1, Hanna Hammock prescribed burn, Tall Timbers (FL), 28 Feb 2023.
# Temperatures are radiometric brightness temperatures in deg C.
SITES = {
    "HH1": {"date": "2023-02-28", "year": 2023, "plot": 1,
            # train/test split(s) exported to the page (snapshots used for training, counted from first_snapshot)
            "k_list": [8]},
}

# Candidate-term library (keys understood by discover.library and the web page).  The wind is one
# constant over the 25-min record, so only its canopy-reduced, spatially varying midflame value (U·cos θ)
# can enter; M, S, R0 come from the model-input layers (pipeline/landscape.py), P is the fire's memory.
# Direction-aware library (final): every input enters through a term that speeds the head more than the back.
# Rothermel's and FBP's per-cell rates (R_Roth, R_FBP) carry fuel model, thermal and satellite moisture, canopy
# wind reduction, slope and the Buildup Index; U·cos θ wind speed; FRFD·cos θ heat flux; κ·cos θ front shape.
TERMS = ["1", "cos", "cos2", "Icos", "kcos", "Ucos", "Rr", "Rf"]
ALL_TERMS = ["1", "cos", "sin", "cos2", "I", "Icos", "k", "kcos", "T", "m", "s", "Ucos", "R0", "P", "Rr", "Rf"]   # ablation


@dataclass
class Config:
    site: str = "HH1"
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    web_dir: Path = field(default_factory=lambda: ROOT / "web")
    epsg: int = 26916                # NAD83 / UTM 16N (the frames' CRS)

    # --- frames / masks / fronts ----------------------------------------------
    grid_res_m: float = 0.5          # analysis grid
    snapshot_s: float = 30.0         # forecast step (frames are every 5 s)
    first_snapshot: int = 23         # analysis starts at the crew ignition (15:57:55); earlier snapshots
                                     # only show the slow north-east front before the head fire
    thr_active: float = 300.0        # deg C: flaming
    thr_delta: float = 120.0         # deg C: "hot" -> burnt (cumulative over every 5 s frame)
    thr_burnt0: float = 60.0         # deg C: already burnt (warm ash) in the first frame
    min_blob_px: int = 6             # drop burnt specks smaller than this
    max_hole_px: int = 40            # fill unburnt holes smaller than this
    contour_sigma_px: float = 1.0    # smoothing before sub-pixel contouring
    curvature_sigma_px: float = 2.0  # smoothing for curvature
    max_advance_m: float = 30.0      # ROS rays longer than this are discarded
    ign_detect_m: float = 12.0       # new fire farther than this from burnt ground = ignition / spot
    ign_exclude_m: float = 4.0       # front points this close to a new ignition are excluded from fitting
    ign_r0_m: float = 0.5            # radius given to ignition sources in the forecast
    edge_exclude_m: float = 2.0      # ... and this close to the plot / fuel boundary
    tile_m: float = 15.0             # spatial block size for the block bootstrap

    # --- models / split -----------------------------------------------------
    train_fire_passes: int = 8       # snapshots used for training, from first_snapshot (train 23-30, test 30->31 ... 49->50)
    stlsq_threshold: float | None = None  # m/s on standardised terms; None = choose by CV on training data
    lambda_criterion: str = "forecast"    # "forecast": held-out new-area IoU of training-interval forecasts; "mse": ROS error
    sign_priors: bool = True              # drop terms whose sign contradicts fire physics (discover.SIGN_PRIOR)
    term_keys: tuple | None = None        # candidate library override (None = TERMS); used by the ablation
    forced_terms: tuple = tuple(TERMS)    # every input stays in the equation (user's choice); () = sparse selection
    ridge_grid: tuple = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # shrinkage chosen by held-out forecast skill on training data
    unpenalized_terms: tuple = ("cos", "cos2")  # the wind-direction law is not shrunk: head / flank / back follow the data
    ridge: float = 1e-3
    n_bootstrap: int = 200

    @classmethod
    def for_site(cls, site: str = "HH1", **kw) -> "Config":
        return cls(site=site, **kw)

    @property
    def kind(self) -> str:
        return "flame3"

    @property
    def terms(self) -> list[str]:
        return list(self.term_keys) if self.term_keys else list(TERMS)

    @property
    def date(self) -> str:
        return SITES[self.site]["date"]

    @property
    def year(self) -> int:
        return SITES[self.site]["year"]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim" / self.site

    @property
    def qa_dir(self) -> Path:
        return self.data_dir / "qa" / self.site

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in d.items()}, indent=2)
