"""Validated offline inputs and bounded, explicit CFPB snapshot ingestion."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
from backend.config import FEATURES, YEARS

FILTERS = {'action_taken': [1, 2, 3], 'loan_purpose': [1], 'lien_status': [1],
           'occupancy_type': [1], 'construction_method': [1], 'total_units': [1, 2, 3, 4]}
DEMOGRAPHICS = ['derived_race', 'derived_ethnicity', 'derived_sex']
REQUIRED = list(dict.fromkeys(['activity_year', *FEATURES, *FILTERS]))
KEEP = list(dict.fromkeys(REQUIRED + DEMOGRAPHICS + ['lei', 'state_code', 'county_code',
                        'census_tract', 'property_value', 'applicant_age']))
SOURCE_DOC = 'https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def source_url(year):
    if year not in YEARS: raise ValueError('Only verified schema years 2021–2025 are supported.')
    return f'https://files.ffiec.cfpb.gov/static-data/snapshot/{year}/{year}_public_lar_csv.zip'


def validate(frame, year, already_filtered=False):
    missing = set(REQUIRED) - set(frame.columns)
    if missing: raise ValueError(f'{year}: missing required fields {sorted(missing)}')
    if not pd.to_numeric(frame.activity_year, errors='coerce').eq(year).all():
        raise ValueError(f'{year}: activity_year mismatch or missing year')
    if 'source_year' in frame and not pd.to_numeric(frame.source_year, errors='coerce').eq(year).all():
        raise ValueError(f'{year}: source_year mismatch')
    mask = pd.Series(True, index=frame.index)
    for column, allowed in FILTERS.items():
        mask &= pd.to_numeric(frame[column], errors='coerce').isin(allowed)
    if already_filtered and not mask.all():
        raise ValueError(f'{year}: mounted sample violates documented cohort filters')
    selected = frame.loc[mask].copy()
    target = pd.to_numeric(selected.action_taken).isin([1, 2]).astype(int)
    if 'approved' in selected and not pd.to_numeric(selected.approved, errors='coerce').eq(target).all():
        raise ValueError(f'{year}: approved label conflicts with actions 1 + 2 = positive, 3 = denied')
    selected['approved'] = target
    selected['source_year'] = year
    return selected


def sample_archive(path, year, size=25000, seed=42, chunksize=100000, max_uncompressed=20 * 1024**3):
    rng = np.random.default_rng(seed)
    sample = None
    scanned = eligible = 0
    with zipfile.ZipFile(path) as archive:
        members = [i for i in archive.infolist() if i.filename.lower().endswith('.csv') and not i.is_dir()]
        if len(members) != 1: raise ValueError('Expected exactly one public LAR CSV in archive')
        member = members[0]
        if member.file_size > max_uncompressed: raise ValueError('Uncompressed archive exceeds limit')
        # Read directly; never extract archive-controlled paths.
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, dtype=str, chunksize=chunksize, usecols=lambda c: c in KEEP):
                scanned += len(chunk)
                selected = validate(chunk, year)
                eligible += len(selected)
                if selected.empty: continue
                selected['_priority'] = rng.random(len(selected))
                sample = selected if sample is None else pd.concat([sample, selected], ignore_index=True)
                sample = sample.nsmallest(size, '_priority')
    if sample is None or sample.empty: raise ValueError(f'{year}: no eligible records')
    return sample.drop(columns='_priority').reset_index(drop=True), {'rows_scanned': scanned, 'eligible_rows': eligible,
           'sample_rows': len(sample), 'sample_seed': seed, 'sampling': 'lowest uniform random priorities across full CSV; fixed order and seed',
           'archive_member': member.filename, 'uncompressed_bytes': member.file_size}


def download(year, destination, max_bytes):
    url = source_url(year)
    with requests.get(url, stream=True, timeout=(20, 120), allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Official CFPB archive returned HTTP {response.status_code} for {year}; use mounted inputs or verified local archives.')
        length = response.headers.get('Content-Length')
        if length and int(length) > max_bytes: raise ValueError('Download exceeds byte budget')
        total = 0
        with destination.open('xb') as stream:
            for block in response.iter_content(1024 * 1024):
                total += len(block)
                if total > max_bytes: raise ValueError('Download exceeds byte budget')
                stream.write(block)
    return {'url': url, 'bytes': total, 'etag': response.headers.get('ETag'),
            'last_modified': response.headers.get('Last-Modified')}


def ingest(run_dir, input_dir, source='mounted', allow_download=False, size=25000, seed=42,
           max_download_bytes=2 * 1024**3):
    if size <= 0 or max_download_bytes <= 0: raise ValueError('Sample size and byte limit must be positive')
    if source not in {'mounted', 'archives', 'cfpb'}: raise ValueError('Unknown input source')
    if source == 'cfpb' and not allow_download:
        raise ValueError('CFPB download requires explicit --allow-download; no network fallback is automatic')
    output = run_dir / 'data'
    output.mkdir()
    origin = input_dir / 'ingestion_manifest_2022_2025.json'
    historical = json.loads(origin.read_text()) if source == 'mounted' and origin.is_file() else {}
    if historical:
        write_json(run_dir / 'upstream_manifest.json', historical)
    results = []
    for year in YEARS:
        details = {'year': year, 'source_mode': source, 'source_url': source_url(year),
                   'release_family': 'snapshot', 'publication_reference': SOURCE_DOC + f'/{year}'}
        if source == 'mounted':
            path = input_dir / f'hmda_{year}.parquet'
            before = digest(path)
            frame = validate(pd.read_parquet(path), year, already_filtered=True)
            details.update(input_path=str(path.resolve()), input_sha256=before, rows_scanned=len(frame),
                           eligible_rows=len(frame), sample_rows=len(frame), sampling='preserved mounted sample without resampling',
                           upstream_provenance=historical.get(str(year)),
                           upstream_verification='Prior manifest/notebook claim only; source archive not available for re-verification')
        else:
            if source == 'cfpb':
                (run_dir / 'downloads').mkdir(exist_ok=True)
                path = run_dir / 'downloads' / f'{year}_public_lar_csv.zip'
                details['download'] = download(year, path, max_download_bytes)
            else:
                path = input_dir / f'{year}_public_lar_csv.zip'
            before = digest(path)
            frame, counts = sample_archive(path, year, size, seed)
            details.update(counts, input_path=str(path.resolve()), input_sha256=before,
                           upstream_verification='Official host download' if source == 'cfpb' else 'User-supplied snapshot archive; URL not independently authenticated')
        if digest(path) != before: raise ValueError(f'{year}: input changed while reading')
        if frame.empty or set(frame.approved.unique()) != {0, 1}:
            raise ValueError(f'{year}: nonempty data with both target classes required')
        target = output / f'hmda_{year}.parquet'
        # The run directory is exclusively created; no existing Dataset is written.
        with target.open('xb') as stream: frame.to_parquet(stream, index=False)
        details.update(output_path=str(target), output_sha256=digest(target), columns=list(frame.columns),
                       class_counts={str(k): int(v) for k, v in frame.approved.value_counts().items()},
                       missing_demographic_fields=[c for c in DEMOGRAPHICS if c not in frame])
        results.append(details)
    manifest = {'filters': FILTERS, 'years': results, 'source_notes': 'See docs/HMDA_SOURCE.md; archive hashes identify exact bytes, URLs alone do not pin a vintage.'}
    write_json(run_dir / 'ingestion.json', manifest)
    return manifest
