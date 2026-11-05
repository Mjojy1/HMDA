"""Managed training runs only. Success requires server values and artifact byte readback."""
import json
import math
import os
from pathlib import Path
import tempfile
from experiments.log_model_comparison import tracking_client
from workflows.hmda_input import digest


class TrainingTracking:
    def __init__(self, client=None):
        self.client, self.version = (client, 'test') if client is not None else tracking_client()
        self.run_id = None

    def start(self, manifest):
        c = self.client
        project = os.environ['DOMINO_PROJECT_ID']
        name = f'HMDA Training - {project}'
        exp = c.get_experiment_by_name(name)
        if exp is None:
            eid = c.create_experiment(name, tags={'domino.project_id': project})
            exp = c.get_experiment(eid)
        if exp.lifecycle_stage != 'active' or exp.tags.get('domino.project_id') != project:
            raise ValueError('Training experiment is inactive or belongs to another project')
        self.experiment_id = exp.experiment_id
        tags = {'mlflow.runName': 'HMDA temporal classifier training', 'hmda.run_type': 'training',
                'domino.project_id': project, 'domino.run_id': os.getenv('DOMINO_RUN_ID', ''),
                'hmda.training_performed': 'false', 'git.commit_sha': manifest['git']['commit_sha'] or 'unknown'}
        self.run_id = c.create_run(exp.experiment_id, tags=tags).info.run_id

    def finish(self, directory, manifest, metadata, evaluation):
        c, rid = self.client, self.run_id
        params = {**{'estimator.' + k: str(v) for k, v in metadata['parameters'].items()},
                  **{k: json.dumps(metadata[k], sort_keys=True) for k in
                     ['features', 'seed', 'training_years', 'validation_year', 'test_year', 'threshold',
                      'threshold_selection', 'preprocessing', 'split_counts']}}
        values = {f'{split}.{k}': v for split, data in
                  [('validation', evaluation['validation_metrics']), ('holdout', evaluation['classification_metrics'])]
                  for k, v in data.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        for k, v in params.items(): c.log_param(rid, k, v)
        for k, v in values.items(): c.log_metric(rid, k, v)
        c.set_tag(rid, 'hmda.training_performed', 'true')
        c.set_tag(rid, 'model.sha256', evaluation['model_sha256'])
        # Raw HMDA input copies stay in Job artifacts. Tracking retains provenance,
        # source code, fitted pipeline and all evaluation reports.
        paths = [p for p in directory.rglob('*') if p.is_file() and p.relative_to(directory).parts[0] not in {'data', 'downloads'}]
        provenance = directory / 'training_provenance.json'
        from workflows.hmda_input import write_json
        write_json(provenance, manifest)
        paths.append(provenance)
        hashes = {}
        for p in paths:
            relative = p.relative_to(directory)
            hashes[str(relative)] = digest(p)
            c.log_artifact(rid, str(p), artifact_path=str(relative.parent) if relative.parent != Path('.') else None)
        c.set_terminated(rid, status='FINISHED')
        run = c.get_run(rid)
        if run.info.status != 'FINISHED' or run.data.tags.get('hmda.run_type') != 'training' or run.data.tags.get('hmda.training_performed') != 'true':
            raise ValueError('Managed training run status/type readback mismatch')
        for k, v in params.items():
            if run.data.params.get(k) != v: raise ValueError(f'Server parameter mismatch: {k}')
        for k, v in values.items():
            if not math.isclose(run.data.metrics.get(k, float('nan')), v, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f'Server metric mismatch: {k}')
        with tempfile.TemporaryDirectory(prefix='hmda-training-readback-') as temp:
            for name, expected in hashes.items():
                if digest(c.download_artifacts(rid, name, temp)) != expected:
                    raise ValueError(f'Server artifact mismatch: {name}')
        return {'status': 'verified', 'run_id': rid, 'experiment_id': self.experiment_id,
                'mlflow_version': self.version, 'metrics': values, 'artifact_sha256': hashes}

    def fail(self):
        if self.run_id: self.client.set_terminated(self.run_id, status='FAILED')
