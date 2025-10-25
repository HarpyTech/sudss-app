#!/usr/bin/env python3
"""
generate_synthetic_minimal_fixedcounts.py

Generates minimal synthetic EHR CSVs with:
 - patients.csv (id, given_name, family_name, age, gender)
 - conditions.csv (>=5 && <=15 rows per patient)
 - imaging_studies.csv (>=5 && <=15 rows per patient; creates PNG placeholders)
 - observations.csv (>=5 && <=15 rows per patient; linked to imaging studies)
 - reports.csv (>=5 && <=15 rows per patient; includes text_preview = findings + impression + evidence lines)
 - history_summary.csv (one row per (patient, code) with baseline/latest/delta/trend/n_points)

Usage:
    python generate_synthetic_minimal_fixedcounts.py --outdir ./synthetic_minimal --n_patients 50 --seed 42

Dependencies:
    pip install pandas numpy faker pillow tqdm
"""
import argparse
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm

fake = Faker()


# ---------------- helpers ----------------
def iso_date(dt):
    return dt.strftime("%Y-%m-%d")


def rand_dates(start, end, n):
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    times = sorted(random.uniform(start_ts, end_ts) for _ in range(n))
    return [datetime.fromtimestamp(t) for t in times]


def make_placeholder(path: Path, size=(512, 512), seed_val=None, modality="XR"):
    if seed_val is not None:
        np.random.seed(seed_val & 0xFFFFFFFF)
        random.seed(seed_val & 0xFFFFFFFF)
    w, h = size
    base = np.linspace(30, 140, h).astype(np.uint8)
    arr = np.tile(base[:, None], (1, w))
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)
    # add a few ellipses to mimic structures
    for _ in range(random.randint(2, 6)):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        r = random.randint(10, min(w, h) // 6)
        shade = random.randint(10, 80)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=shade)
    # modality variation
    if modality == "CT":
        img = img.filter(ImageFilter.GaussianBlur(radius=1.6))
    elif modality == "MRI":
        img = img.filter(ImageFilter.GaussianBlur(radius=2.2))
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
    # add noise
    np_img = np.array(img).astype(np.int16)
    noise = (np.random.normal(0, 6, np_img.shape)).astype(np.int16)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    Image.fromarray(np_img).save(path, format="PNG", compress_level=1)


# ---------------- templates / choices ----------------
OBS_TEMPLATES = [
    {
        "code": "BNP",
        "desc": "Brain natriuretic peptide",
        "units": "pg/mL",
        "typical": 400,
    },
    {"code": "O2SAT", "desc": "Oxygen saturation", "units": "%", "typical": 96},
    {"code": "HR", "desc": "Heart rate", "units": "bpm", "typical": 78},
    {
        "code": "CXR_SCORE",
        "desc": "CXR severity score (0-3)",
        "units": "score",
        "typical": 1,
    },
    {"code": "WBC", "desc": "White blood cell count", "units": "10^3/uL", "typical": 8},
]

CONDITIONS = [
    ("I50.9", "Heart failure, unspecified"),
    ("J18.9", "Pneumonia, unspecified"),
    ("I10", "Hypertension"),
    ("E11.9", "Type 2 diabetes"),
    ("R09.89", "Respiratory symptom"),
]

MODALITIES = ["XR", "CT", "MRI", "DIGITAL"]

FINDINGS_POOL = [
    "No acute cardiopulmonary disease identified.",
    "Cardiomegaly with small bilateral pleural effusions.",
    "Lobar consolidation in right lower lobe consistent with pneumonia.",
    "Chronic interstitial changes without acute airspace disease.",
    "Mild pulmonary edema.",
]

IMPRESSION_POOL = [
    "Findings consistent with pulmonary edema and cardiomegaly.",
    "Findings suspicious for lobar pneumonia.",
    "No acute cardiopulmonary process identified.",
    "Chronic changes; correlate clinically.",
]


# ---------------- main ----------------
def generate(
    outdir: Path,
    n_patients: int,
    seed: int,
    min_per_patient: int = 5,
    max_per_patient: int = 15,
    img_size=(512, 512),
):
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    outdir.mkdir(parents=True, exist_ok=True)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    patients_rows = []
    conditions_rows = []
    imaging_rows = []
    observations_rows = []
    reports_rows = []
    history_rows = []

    now = datetime.utcnow()
    start_global = now - timedelta(days=365 * 3)

    # We'll store per-patient, per-code time series to compute history_summary
    per_patient_series = {}  # key (pid, code) -> list of (date,value)

    for pidx in tqdm(range(n_patients), desc="Generating patients"):
        pid = f"P{100000 + pidx}"
        name_parts = fake.name().split()
        given = name_parts[0]
        family = name_parts[-1] if len(name_parts) > 1 else ""
        birth = fake.date_of_birth(minimum_age=30, maximum_age=90)
        age = now.year - birth.year - ((now.month, now.day) < (birth.month, birth.day))
        gender = random.choice(["male", "female", "other"])
        patients_rows.append(
            {
                "patient_id": pid,
                "given_name": given,
                "family_name": family,
                "age": age,
                "gender": gender,
            }
        )

        # For each non-patient file choose a random count between min & max
        n_conditions = random.randint(min_per_patient, max_per_patient)
        n_imaging = random.randint(min_per_patient, max_per_patient)
        n_observations = random.randint(min_per_patient, max_per_patient)
        n_reports = random.randint(min_per_patient, max_per_patient)
        # We will link imaging <-> observations <-> reports where possible. We'll create imaging_count images,
        # then for observations and reports we attach them to imaging events in round-robin fashion.

        # conditions: produce n_conditions consultation rows
        cond_dates = rand_dates(start_global, now, n_conditions)
        for dt in cond_dates:
            code, desc = random.choice(CONDITIONS)
            conditions_rows.append(
                {
                    "patient_id": pid,
                    "consultation_date": iso_date(dt),
                    "condition_code": code,
                    "condition_description": desc,
                }
            )

        # imaging studies (n_imaging)
        img_dates = rand_dates(start_global, now, n_imaging)
        for ii, dt in enumerate(img_dates):
            modality = random.choice(MODALITIES)
            series_uid = str(uuid.uuid4())
            instance_uid = str(uuid.uuid4())
            imaging_id = f"IMG-{pid}-{ii}"
            fname = f"{instance_uid}.png"
            image_path = str(img_dir / fname)
            # create placeholder image
            make_placeholder(
                Path(image_path),
                size=img_size,
                seed_val=hash(instance_uid),
                modality=modality,
            )
            imaging_rows.append(
                {
                    "imaging_id": imaging_id,
                    "series_uid": series_uid,
                    "instance_uid": instance_uid,
                    "patient_id": pid,
                    "date": iso_date(dt),
                    "modality": modality,
                    "image_path": image_path,
                }
            )

        # observations: create n_observations rows; attach each to an imaging event (round-robin)
        # We will ensure at least one observation per imaging; if n_observations > n_imaging multiple observations per imaging are allowed
        for oi in range(n_observations):
            obs_dt = random.choice(
                img_dates
            )  # associate with an imaging date for evidence proximity
            imaging_idx = oi % len(img_dates)
            imaging_id = f"IMG-{pid}-{imaging_idx}"
            instance_uid = imaging_rows[imaging_idx]["instance_uid"]
            ot = random.choice(OBS_TEMPLATES)
            # generate or extend per-patient per-code series
            key = (pid, ot["code"])
            if key not in per_patient_series:
                # produce a small sequence across the number of imaging events for this patient
                base = ot["typical"] * random.uniform(0.85, 1.2)
                direction = random.choices(
                    ["improve", "worsen", "stable"], weights=[0.35, 0.35, 0.3]
                )[0]
                seq = []
                for kstep in range(len(img_dates)):
                    jitter = np.random.normal(scale=0.04 * base)
                    if direction == "improve":
                        val = base - (kstep * base * 0.04) + jitter
                    elif direction == "worsen":
                        val = base + (kstep * base * 0.04) + jitter
                    else:
                        val = base + jitter
                    seq.append(round(max(0, float(val)), 2))
                per_patient_series[key] = {
                    "seq": seq,
                    "direction": direction,
                    "dates": [d for d in img_dates],
                    "units": ot["units"],
                }
            seq_info = per_patient_series[key]
            # pick value corresponding to this imaging index
            idx_for_imaging = imaging_idx if imaging_idx < len(seq_info["seq"]) else -1
            val = seq_info["seq"][idx_for_imaging]
            observations_rows.append(
                {
                    "patient_id": pid,
                    "imaging_id": imaging_id,
                    "modality": imaging_rows[imaging_idx]["modality"],
                    "instance_uid": instance_uid,
                    "date": iso_date(obs_dt),
                    "code": ot["code"],
                    "description": ot["desc"],
                    "value": val,
                    "units": ot["units"],
                    "type": (
                        "lab" if ot["code"] not in ("CXR_SCORE",) else "imaging-derived"
                    ),
                }
            )

        # reports: create n_reports; attach to imaging in round-robin; include text_preview (findings+impression+evidence)
        for ri in range(n_reports):
            imaging_idx = ri % len(img_dates)
            imaging_id = f"IMG-{pid}-{imaging_idx}"
            img_dt = img_dates[imaging_idx]
            findings = random.choice(FINDINGS_POOL)
            impression = random.choice(IMPRESSION_POOL)
            # build short evidence lines: choose up to 3 observations for this imaging (by filtering observations_rows)
            evidence_lines = []
            # gather obs for this patient & imaging
            obs_candidates = [
                o
                for o in observations_rows
                if o["patient_id"] == pid and o["imaging_id"] == imaging_id
            ]
            # if none found (unlikely), sample some from per_patient_series
            if not obs_candidates:
                # create a synthetic candidate line
                for k in range(min(2, len(list(per_patient_series.keys())))):
                    key = list(per_patient_series.keys())[k]
                    code = key[1]
                    val = per_patient_series[key]["seq"][0]
                    evidence_lines.append(
                        f"{code}: {val} {per_patient_series[key]['units']}"
                    )
            else:
                # pick up to 3 obs and format
                cho = random.sample(obs_candidates, k=min(3, len(obs_candidates)))
                for o in cho:
                    evidence_lines.append(
                        f"{o['code']}={o['value']}{o.get('units','')} ({o['date']})"
                    )
            # text_preview concatenates findings, impression, and evidence lines
            text_preview = findings + " " + impression
            if evidence_lines:
                text_preview += " Evidence: " + "; ".join(evidence_lines)
            report_id = f"RPT-{pid}-{ri}"
            reports_rows.append(
                {
                    "report_id": report_id,
                    "patient_id": pid,
                    "imaging_id": imaging_id,
                    "date": iso_date(img_dt),
                    "findings": findings,
                    "impression": impression,
                    "summary": impression if random.random() < 0.8 else findings,
                    "text_preview": text_preview,
                }
            )

    # After all patients: build history_summary rows
    for (pid, code), info in per_patient_series.items():
        seq = info["seq"]
        baseline = seq[0]
        latest = seq[-1]
        delta = round(latest - baseline, 3)
        if abs(delta) < 0.02 * (baseline + 1e-6):
            trend = "stable"
        else:
            trend = "improved" if delta < 0 else "worsened"
        history_rows.append(
            {
                "patient_id": pid,
                "code": code,
                "baseline_value": baseline,
                "latest_value": latest,
                "delta": delta,
                "trend": trend,
                "n_points": len(seq),
                "units": info.get("units", ""),
            }
        )

    # Save CSVs
    pd.DataFrame(patients_rows).to_csv(outdir / "patients.csv", index=False)
    pd.DataFrame(conditions_rows).to_csv(outdir / "conditions.csv", index=False)
    pd.DataFrame(imaging_rows).to_csv(outdir / "imaging_studies.csv", index=False)
    pd.DataFrame(observations_rows).to_csv(outdir / "observations.csv", index=False)
    pd.DataFrame(reports_rows).to_csv(outdir / "reports.csv", index=False)
    pd.DataFrame(history_rows).to_csv(outdir / "history_summary.csv", index=False)

    print(f"Saved dataset to {outdir}")
    print("Counts:")
    print(" patients:", len(patients_rows))
    print(" conditions:", len(conditions_rows))
    print(" imaging:", len(imaging_rows))
    print(" observations:", len(observations_rows))
    print(" reports:", len(reports_rows))
    print(" history rows:", len(history_rows))
    print(" image files saved to:", img_dir)


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, default="./synthetic_minimal", help="Output directory"
    )
    parser.add_argument(
        "--n_patients", type=int, default=50, help="Number of synthetic patients"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min_per_patient",
        type=int,
        default=5,
        help="Minimum rows per patient for each file (except patients.csv)",
    )
    parser.add_argument(
        "--max_per_patient",
        type=int,
        default=15,
        help="Maximum rows per patient for each file (except patients.csv)",
    )
    args = parser.parse_args()

    generate(
        Path(args.outdir),
        n_patients=args.n_patients,
        seed=args.seed,
        min_per_patient=args.min_per_patient,
        max_per_patient=args.max_per_patient,
    )
