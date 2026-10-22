"""Prepare the real 2024 ACS California wage sample; never fits a regression.

Run with this project's Python: datasets/prepare_acs.py [--download-only].
The default matrix is UNWEIGHTED. PWGTP is saved separately; PUMA clustering
is a model-based geographic illustration, not ACS survey-design inference.
Raw files and generated arrays are deliberately kept outside release sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SOURCES = {
    "csv_pca.zip": "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pca.zip",
    "PUMS_Data_Dictionary_2024.txt": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt",
    "2020_Census_Tract_to_2020_PUMA.txt": "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt",
}
NUMERIC_COLUMNS = ["AGEP", "COW", "ESR", "WAGP", "WKWN", "WKHP", "ADJINC",
                   "SCHL", "SEX", "RAC1P", "HISP", "NATIVITY", "MAR", "PUMA",
                   "OCCP", "INDP", "PWGTP", "FWAGP", "FWKHP", "FWKWNP"]
EDUCATION_LEVELS = ["less_than_hs", "high_school", "some_college", "associate",
                    "bachelor", "postgraduate"]
# An explicitly chosen proxy mapping, not an ACS measure of completed years.
SCHOOL_YEARS = {1: 0, 2: 0, 3: 0, **{k: k - 3 for k in range(4, 15)},
                15: 12, 16: 12, 17: 12, 18: 13, 19: 14, 20: 14,
                21: 16, 22: 18, 23: 19, 24: 20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_sources(raw: Path, *, no_download: bool) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE / "source_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {}
    for name, url in SOURCES.items():
        target = raw / name
        downloaded = False
        if not target.exists():
            if no_download:
                raise FileNotFoundError(f"Missing {target}; rerun without --no-download")
            print(f"Downloading official Census source: {name}", flush=True)
            temporary = target.with_name(target.name + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "ACS-QR-reproducible-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temporary.replace(target)
            downloaded = True
        digest = sha256(target)
        old = previous.get(name, {})
        if old.get("sha256") and old["sha256"] != digest:
            raise ValueError(f"Source hash differs from recorded source: {name}")
        manifest[name] = {"url": url, "sha256": digest, "bytes": target.stat().st_size,
                          "retrieved_utc": old.get("retrieved_utc") or datetime.now(timezone.utc).isoformat(),
                          "local_file": f"raw/{name}"}
        # Save after each source so interrupted downloads preserve provenance.
        merged = {**previous, **manifest}
        manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"{'Downloaded' if downloaded else 'Verified'} {name}: {digest}", flush=True)
    return manifest


def occupation_industry_maps(dictionary: Path) -> tuple[dict, dict, dict]:
    text = dictionary.read_text(encoding="utf-8-sig")
    maps, descriptions = {}, {}
    for variable in ("OCCP", "INDP"):
        match = re.search(rf"^{variable}\s+Character\s+4\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing official dictionary section {variable}")
        rest = text[match.end():]
        next_header = re.search(r"^[A-Z][A-Z0-9]*\s+(?:Character|Numeric)\s+\d+", rest, re.MULTILINE)
        section = rest[:next_header.start()] if next_header else rest
        codes = {}
        for line in section.splitlines():
            found = re.match(r"\s*(\d{4})\s+\.([A-Z]+)-(.+)", line)
            if found:
                code, group, label = found.groups()
                codes[int(code)] = group
                descriptions[f"{variable}:{code}"] = f"{group}-{label.strip()}"
        if len(codes) < 100:
            raise ValueError(f"Unexpectedly few official {variable} codes: {len(codes)}")
        maps[variable] = codes
    return maps["OCCP"], maps["INDP"], descriptions


def county_components(crosswalk: Path) -> tuple[dict, dict]:
    """Coarsen geography, never impute a person's unobserved county.

    Join counties if one published PUMA contains tracts in both. Connected
    components are county unions containing whole PUMAs. Their IDs are sorted
    county FIPS codes; these are an analyst-defined coarsening, not new official
    geography. Tract-level source information is used only for this crosswalk.
    """
    cross = pd.read_csv(crosswalk, dtype=str, encoding="utf-8-sig")
    cross = cross.loc[cross["STATEFP"] == "06"]
    parent = {code: code for code in cross["COUNTYFP"].unique()}

    def find(code):
        while parent[code] != code:
            parent[code] = parent[parent[code]]
            code = parent[code]
        return code

    puma_counties = cross.groupby("PUMA5CE", sort=True)["COUNTYFP"].unique()
    for members in puma_counties:
        first = members[0]
        for other in members[1:]:
            left, right = sorted((find(first), find(other)))
            parent[right] = left
    groups = {}
    for county in sorted(parent):
        groups.setdefault(find(county), []).append(county)
    names = {key: "+".join(value) for key, value in groups.items()}
    puma_to_group = {int(puma): names[find(members[0])] for puma, members in puma_counties.items()}
    return puma_to_group, {names[key]: value for key, value in groups.items()}


def read_sample(archive: Path, occ_map: dict, ind_map: dict, geography: dict):
    parts, counts = [], Counter()
    members_metadata = []
    offset = 0
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(name for name in zipped.namelist()
                         if name.lower().endswith(".csv") and Path(name).name.lower().startswith("psam_p"))
        if not members:
            raise ValueError("No ACS person CSV member found")
        for member in members:
            with zipped.open(member) as source:
                headers = pd.read_csv(source, nrows=0).columns.tolist()
            state = "STATE" if "STATE" in headers else "ST"
            columns = NUMERIC_COLUMNS + [state, "SERIALNO", "SPORDER"]
            absent = sorted(set(columns) - set(headers))
            if absent:
                raise ValueError(f"Missing required ACS columns: {absent}")
            members_metadata.append({"name": member, "uncompressed_bytes": zipped.getinfo(member).file_size,
                                     "crc32": f"{zipped.getinfo(member).CRC:08x}"})
            with zipped.open(member) as source:
                for chunk in pd.read_csv(source, usecols=columns, chunksize=50000,
                                         dtype={"SERIALNO": "string"}, low_memory=False):
                    chunk["source_row"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
                    offset += len(chunk)
                    counts["raw_person_records"] += len(chunk)
                    chunk.rename(columns={state: "STATE"}, inplace=True)
                    for column in NUMERIC_COLUMNS + ["STATE", "SPORDER"]:
                        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                    mask = pd.Series(True, index=chunk.index)
                    steps = [
                        ("california", chunk["STATE"].eq(6)),
                        ("age_25_64", chunk["AGEP"].between(25, 64)),
                        ("civilian_currently_employed", chunk["ESR"].isin([1, 2])),
                        ("wage_salary_class_1_to_5", chunk["COW"].between(1, 5)),
                        ("positive_annual_wages", chunk["WAGP"].gt(0)),
                        ("weeks_1_to_52", chunk["WKWN"].between(1, 52)),
                        ("hours_1_to_98_excludes_99plus", chunk["WKHP"].between(1, 98)),
                        ("valid_adjustment_and_weight", chunk["ADJINC"].gt(0) & chunk["PWGTP"].gt(0)),
                        ("valid_covariates", chunk["SCHL"].between(1, 24) & chunk["SEX"].isin([1, 2]) &
                         chunk["RAC1P"].between(1, 9) & chunk["HISP"].between(1, 24) &
                         chunk["NATIVITY"].isin([1, 2]) & chunk["MAR"].between(1, 5) &
                         chunk["OCCP"].isin(occ_map) & chunk["INDP"].isin(ind_map) &
                         chunk["PUMA"].isin(geography)),
                    ]
                    for name, condition in steps:
                        mask &= condition
                        counts[name] += int(mask.sum())
                    parts.append(chunk.loc[mask].copy())
    sample = pd.concat(parts, ignore_index=True)
    if sample.empty:
        raise ValueError("Prespecified sample is empty")
    if sample.duplicated(["SERIALNO", "SPORDER"]).any():
        raise ValueError("Duplicate person identifier in official source; refusing replication")
    return sample, dict(counts), members_metadata


def create_design(sample: pd.DataFrame, occ_map, ind_map, geography):
    n = len(sample)
    values, columns, factors = [np.ones(n)], ["intercept"], {}

    def add(name, value):
        values.append(np.asarray(value, dtype=np.float64))
        columns.append(name)

    def dummies(name, value, reference=None, levels=None):
        value = np.asarray(value).astype(str)
        unique, frequencies = np.unique(value, return_counts=True)
        observed = dict(zip(unique.tolist(), frequencies.astype(int).tolist()))
        if levels is None:
            levels = unique.tolist()
        levels = [level for level in levels if level in observed]
        if reference is None:
            reference = min(observed, key=lambda key: (-observed[key], key))
        if reference not in observed:
            raise ValueError(f"Reference {reference} absent for {name}")
        factors[name] = {"reference": reference, "levels": levels, "counts": observed}
        for level in levels:
            if level != reference:
                add(f"{name}[{level}]", value == level)
        return value

    female = sample["SEX"].eq(2).to_numpy(dtype=float)
    add("female", female)
    school = sample["SCHL"].to_numpy(dtype=int)
    edu = np.select([school <= 15, school <= 17, school <= 19, school == 20,
                     school == 21], EDUCATION_LEVELS[:-1], default="postgraduate")
    dummies("education", edu, reference="high_school", levels=EDUCATION_LEVELS)
    for level in EDUCATION_LEVELS:
        if level != "high_school":
            add(f"female:education[{level}]", female * (edu == level))
    schooling = sample["SCHL"].map(SCHOOL_YEARS).to_numpy(dtype=float)
    experience = np.maximum(sample["AGEP"].to_numpy() - schooling - 6, 0)
    z = 2 * experience / 60 - 1
    # Legendre basis spans the ordinary fourth-degree experience polynomial,
    # with considerably more moderate column scales than raw powers of years.
    polynomial = np.polynomial.legendre.legvander(z, 4)[:, 1:]
    for degree in range(1, 5):
        add(f"potential_experience_L{degree}", polynomial[:, degree - 1])
    for degree in (1, 2):
        add(f"female:potential_experience_L{degree}", female * polynomial[:, degree - 1])
    dummies("race_code", sample["RAC1P"].astype(int), reference="1")
    add("hispanic", sample["HISP"].gt(1))
    add("foreign_born", sample["NATIVITY"].eq(2))
    add("married", sample["MAR"].eq(1))
    dummies("class_of_worker", sample["COW"].astype(int), reference="1")
    occ = sample["OCCP"].map(occ_map)
    ind = sample["INDP"].map(ind_map)
    geo = sample["PUMA"].map(geography)
    dummies("occupation_group", occ)
    dummies("industry_group", ind)
    dummies("county_union", geo)
    x = np.ascontiguousarray(np.column_stack(values), dtype=np.float64)
    annual = sample["WAGP"].to_numpy() * sample["ADJINC"].to_numpy() / 1_000_000
    hourly = annual / (sample["WKWN"].to_numpy() * sample["WKHP"].to_numpy())
    y = np.ascontiguousarray(np.log(hourly), dtype=np.float64)
    raw_cluster = 600000 + sample["PUMA"].to_numpy(dtype=np.int64)
    cluster_values, cluster = np.unique(raw_cluster, return_inverse=True)
    return x, y, cluster.astype(np.int32), cluster_values, columns, factors, hourly, experience


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    raw = HERE / "raw"
    manifest = fetch_sources(raw, no_download=args.no_download)
    if args.download_only:
        return
    occ_map, ind_map, code_labels = occupation_industry_maps(raw / "PUMS_Data_Dictionary_2024.txt")
    geography, geography_members = county_components(raw / "2020_Census_Tract_to_2020_PUMA.txt")
    sample, flow, members = read_sample(raw / "csv_pca.zip", occ_map, ind_map, geography)
    x, y, cluster, cluster_values, columns, factors, hourly, experience = create_design(
        sample, occ_map, ind_map, geography)
    # Lightweight design verification only; no regression, covariance fit, or
    # bootstrap. Sparse accumulation avoids a dense n*p*p Gram product.
    with threadpool_limits(1):
        sx = sparse.csr_matrix(x)
        gram = (sx.T @ sx).toarray()
        eig = np.linalg.eigvalsh(gram)
    rank_tolerance = eig[-1] * max(x.shape) * np.finfo(float).eps
    rank = int((eig > rank_tolerance).sum())
    if rank != x.shape[1]:
        raise ValueError(f"Design rank check failed: {rank}/{x.shape[1]}; inspect rather than silently drop terms")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite analysis matrix")
    destination = HERE / "processed" / "acs2024_ca_wages"
    destination.mkdir(parents=True, exist_ok=True)
    # Existing RcppCNPy/default npyLoad pilot expects numeric doubles. Integer
    # IDs 0..280 are exact in float64; Python/GPU callers cast to int64 for indexing.
    outputs = {"x": x, "y": y, "cluster": cluster.astype(np.float64),
               "cluster_values": cluster_values,
               "person_weight": sample["PWGTP"].to_numpy(dtype=np.float64),
               "source_row": sample["source_row"].to_numpy(dtype=np.int64),
               "potential_experience": experience.astype(np.float64),
               "income_allocated": sample["FWAGP"].to_numpy(dtype=np.int8),
               "hours_allocated": sample["FWKHP"].to_numpy(dtype=np.int8),
               "weeks_allocated": sample["FWKWNP"].to_numpy(dtype=np.int8)}
    saved = {}
    for name, array in outputs.items():
        path = destination / (name + ".npy")
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
        saved[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256(path)}
    size = np.bincount(cluster)
    meta = {"dataset": "2024 ACS 1-year California person PUMS",
            "n": len(y), "p": x.shape[1], "g": len(cluster_values),
            "tau_grid_proposed": [.1, .25, .5, .75, .9], "columns": columns,
            "response": "log(WAGP * ADJINC / 1e6 / (WKWN * WKHP))",
            "weighting": "unweighted; PWGTP saved separately and NOT applied to x or y",
            "estimand": "conditional wage-proxy associations in retained public-use records; no causal claim",
            "cluster_assumption": "model-based independence across PUMAs; not ACS design variance",
            "cluster_encoding": "float64 exact integer IDs, zero-based index into cluster_values; raw key = 6*100000 + PUMA",
            "sample_flow": flow, "factors": factors, "source_zip_members": members,
            "geography_county_unions": geography_members,
            "puma_to_county_union": {str(k): v for k, v in sorted(geography.items())},
            "schooling_years_proxy": SCHOOL_YEARS,
            "potential_experience": "max(AGEP - schooling_years_proxy - 6, 0); z=2*experience/60-1; Legendre 1:4",
            "design_rank": rank, "rank_threshold": float(rank_tolerance),
            "gram_eigenvalue_min": float(eig[0]), "gram_eigenvalue_max": float(eig[-1]),
            "cluster_size_min_median_max": [int(size.min()), float(np.median(size)), int(size.max())],
            "hourly_proxy_quantiles": {str(q): float(np.quantile(hourly, q)) for q in [0, .01, .1, .5, .9, .99, 1]},
            "hourly_proxy_below_1": int((hourly < 1).sum()), "hourly_proxy_above_1000": int((hourly > 1000).sum()),
            "retained_allocation_rates": {key: float(sample[key].mean()) for key in ["FWAGP", "FWKHP", "FWKWNP"]},
            "unique_annual_wage_values": int(sample["WAGP"].nunique()),
            "unique_hourly_proxy_values": int(np.unique(hourly).size),
            "source_manifest": manifest, "outputs": saved,
            "script_sha256": sha256(Path(__file__)),
            "software": {"numpy": np.__version__, "pandas": pd.__version__},
            "raw_data_in_release_archive": False}
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (destination / "columns.json").write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")
    (destination / "code_labels.json").write_text(json.dumps(code_labels, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "n": len(y), "p": x.shape[1],
                      "g": len(cluster_values), "design_rank": rank,
                      "sample_flow": flow, "cluster_size_min_median_max": meta["cluster_size_min_median_max"]}, indent=2))


if __name__ == "__main__":
    main()
