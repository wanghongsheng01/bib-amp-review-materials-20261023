#!/usr/bin/env python3
"""One receipt-bound, zero-training recovery after an unclassified interruption.

No model is imported or loaded until scope, plan, receipt and input hashes pass.
This is a new generation attempt; the original v2 outcome remains incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
CONTRACT = ROOT / 'protocol/CE_CHECKPOINT_RECOVERY_CONTRACT_v1.json'
PLAN = ROOT / 'protocol/CE_CHECKPOINT_RECOVERY_PLAN_v1.json'
RECEIPT = ROOT / 'review/independent/CE_CHECKPOINT_RECOVERY_PREFLIGHT_RECEIPT_v1.json'
OUTPUT = ROOT / 'runs/ce_checkpoint_recovery_v1'
SOURCE = ROOT / 'runs/formal_v2/categorical_ce'
SOURCE_SUPERVISOR = ROOT / 'runs/single_ce_execution_v2'
FROZEN_RUNNER = ROOT / 'implementation/vae_pipeline_formal_v2.py'
SCOPE = 'AMP_H5_CE_CHECKPOINT_RECOVERY_V1'


class RecoveryInterrupt(BaseException):
    """Bypass the inherited generate() catch(Exception) at the outer deadline."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def exact_path(value, expected):
    p = Path(value)
    if not p.is_absolute() or p.is_symlink() or str(p) != str(p.resolve()) or p != expected:
        raise ValueError('absolute canonical scoped path required: ' + str(expected))
    return p


def verify_bindings(bindings):
    errors = []
    for item in bindings:
        p = Path(item['path'])
        try:
            if not p.is_absolute() or p.is_symlink() or str(p) != str(p.resolve()):
                raise ValueError('noncanonical or symlink path')
            if sha(p) != item['sha256']:
                raise ValueError('SHA256 mismatch')
            if 'bytes' in item and p.stat().st_size != item['bytes']:
                raise ValueError('size mismatch')
        except Exception as exc:
            errors.append({'path': str(p), 'error': repr(exc)})
    return errors


def verify_original_snapshot(contract):
    items = contract['original_25_file_snapshot']
    actual = {str(p) for directory in (SOURCE, SOURCE_SUPERVISOR) for p in directory.iterdir() if p.is_file()}
    expected = {x['path'] for x in items}
    errors = verify_bindings(items)
    if len(items) != 25 or len(expected) != 25 or actual != expected:
        errors.append({'path': 'original snapshot', 'error': 'original file membership differs'})
    return errors


def validate_request(contract_path, plan_path, receipt_path, output):
    exact_path(contract_path, CONTRACT); exact_path(plan_path, PLAN)
    exact_path(receipt_path, RECEIPT); exact_path(output, OUTPUT)
    if OUTPUT.exists():
        raise ValueError('recovery output must not exist; no overwrite or resume')
    contract = read_json(CONTRACT)
    if contract['scope_id'] != SCOPE or contract['recovery_script_sha256'] != sha(__file__):
        raise ValueError('contract scope or implementation binding differs')
    for key, expected in [('original_run_directory', SOURCE), ('original_supervisor_directory', SOURCE_SUPERVISOR),
                          ('output_directory', OUTPUT), ('frozen_runner_path', FROZEN_RUNNER)]:
        exact_path(contract[key], expected)
    exact_path(contract['supervisor_output_directory'], ROOT / 'runs/ce_checkpoint_recovery_execution_v1')
    if contract['budget_seconds'] != {'training_updates': 0, 'generation': 1800, 'outer_total': 1980}:
        raise ValueError('frozen recovery budget differs')
    if contract['generation_requests'] != {'FREE': 16, 'AMD': 16} or contract['sampling_seeds'] != [422, 423]:
        raise ValueError('frozen request scope differs')
    plan = read_json(PLAN); receipt = read_json(RECEIPT)
    if receipt.get('authorized') is not True or receipt.get('scope_id') != SCOPE:
        raise PermissionError('an independent scoped execution receipt is required')
    if receipt.get('plan_sha256') != sha(PLAN) or receipt.get('contract_sha256') != sha(CONTRACT):
        raise PermissionError('receipt does not bind this plan and contract')
    if receipt.get('recovery_script_sha256') != sha(__file__):
        raise PermissionError('receipt does not bind this implementation')
    if plan.get('scope_id') != SCOPE or plan.get('root_execution_authorized') is not True:
        raise PermissionError('explicit root execution authorization is required in the final plan')
    exact_path(plan['workspace'], ROOT.parent)
    if plan.get('contract_sha256') != sha(CONTRACT):
        raise ValueError('final plan contract binding differs')
    supervisor = Path(plan['supervisor_path'])
    if not supervisor.is_absolute() or supervisor.is_symlink() or str(supervisor) != str(supervisor.resolve()) or supervisor.parent != ROOT / 'implementation':
        raise ValueError('supervisor must be an absolute reviewed implementation file')
    if receipt.get('supervisor_sha256') != sha(supervisor):
        raise PermissionError('supervisor receipt binding differs')
    if len(plan.get('jobs', [])) != 1:
        raise ValueError('exactly one recovery job permitted')
    job = plan['jobs'][0]
    argv = [contract['python_executable'], '-u', '-B', str(Path(__file__).resolve()),
            '--contract', str(CONTRACT), '--plan', str(PLAN), '--receipt', str(RECEIPT), '--output', str(OUTPUT)]
    if job.get('id') != 'ce_checkpoint_recovery_v1' or job.get('argv') != argv or job.get('outer_timeout_seconds') != 1980:
        raise ValueError('final plan job, invocation or outer cap differs')
    exact_path(plan['output_root'], Path(contract['supervisor_output_directory']))
    bound = {b['path']: b['sha256'] for b in plan.get('bindings', [])}
    for p in (CONTRACT, Path(__file__).resolve()):
        if bound.get(str(p)) != sha(p):
            raise ValueError('final plan lacks required exact binding: ' + str(p))
    lock = Path(contract['process_lock_path'])
    exact_path(lock, ROOT.parent / 'H5_PROCESS.lock')
    if lock.read_text().strip() != str(os.getppid()):
        raise PermissionError('launch only as child of the serial supervisor holding the task lock')
    errors = verify_bindings(plan['bindings']) + verify_bindings(contract['core_bindings']) + verify_original_snapshot(contract)
    if errors:
        raise ValueError('preflight input identity failure: ' + json.dumps(errors))
    return contract, plan, receipt


def interruption_signal(signum, frame):
    raise RecoveryInterrupt('recovery interrupted by signal ' + str(signum) + '; no retry')


def execute(contract_path, plan_path, receipt_path, output):
    process_started = time.perf_counter()
    contract, plan, receipt = validate_request(contract_path, plan_path, receipt_path, output)
    # All authorization checks precede module import, output creation and weight load.
    config = read_json(SOURCE / 'config.json')
    spec = importlib.util.spec_from_file_location('h5_frozen_formal_v2_recovery', FROZEN_RUNNER)
    m = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(m)
    m.validate_config(config)
    if config['arm'] != 'categorical_ce' or config['generation_seconds_cap'] != 1800 or config['sampling_seeds'] != [422, 423] or config['samples_per_terminal'] != 16 or config['noise_steps'] != 999:
        raise ValueError('original generation contract differs')
    for role in ('train', 'dev'):
        for key in (role, role + '_cohort'):
            if config[key] != contract['allowed_export_source_paths'][key]:
                raise ValueError('unregistered source data path: ' + key)
    checkpoint_meta = {}
    for name in ('vae_state', 'ddpm_ema_state'):
        meta = read_json(SOURCE / (name + '.json'))
        if not (meta['updates'] == meta['requested_updates'] == 1000 and meta['fixed_endpoint_reached'] is True):
            raise ValueError('checkpoint is not the frozen 1000 update endpoint')
        if sha(SOURCE / (name + '.pt')) != meta['sha256'] or meta['tensor_state_sha256'] != contract['checkpoint_tensor_digests'][name]:
            raise ValueError('checkpoint file or expected tensor digest differs')
        checkpoint_meta[name] = meta
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    state = {'state': 'running_recovery_attempt', 'scope_id': SCOPE, 'attempt_id': 'ce_checkpoint_recovery_v1',
        'training_updates_this_attempt': 0, 'original_v2_outcome': 'INCOMPLETE_EXIT_CAUSE_UNKNOWN',
        'original_v2_engineering_gate': 'NOT_EVALUABLE_INTERRUPTION', 'PROMOTE': 'HOLD',
        'recovery_contract_sha256': sha(CONTRACT), 'recovery_plan_sha256': sha(PLAN),
        'independent_receipt_sha256': sha(RECEIPT), 'script_sha256': sha(__file__),
        'protocol_deviation': contract['protocol_deviation'],
        'generation_restart': 'complete original seed422/423 chains; not t200 continuation',
        'source_train_read_attempted': False, 'source_train_read_validated': False,
        'source_dev_read_attempted': False, 'source_dev_read_validated': False, 'test_read': False,
        'budget_seconds': contract['budget_seconds'], 'phase': 'setup', 'model_calls_authorized_only_by_new_receipt': True}
    dump(out / 'config.json', config)
    dump(out / 'recovery_contract.json', contract)
    dump(out / 'phase_provenance.json', contract['phase_provenance'])
    rows = m.allocated_requests(config)
    m.write_csv(out / 'generated.csv', rows, m.GENERATED_FIELDS)
    dump(out / 'status.json', state)
    models = {}; before = {}; generation = None
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    for s in previous_handlers:
        signal.signal(s, interruption_signal)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 1980 - (time.perf_counter() - process_started)))
    try:
        runtime = m.init_runtime(config['source_root'])
        old_runtime = read_json(SOURCE / 'runtime_and_source.json')
        for key in ('torch', 'numpy', 'python', 'threads', 'interop_threads', 'device', 'source_manifest_sha256'):
            if runtime[key] != old_runtime[key]:
                raise ValueError('runtime identity differs: ' + key)
        dump(out / 'runtime_and_source.json', runtime)
        models['vae_state'] = m.make_model(config['arm'], config['initialization_seed'])
        models['vae_state'].load_state_dict(m.torch.load(SOURCE / 'vae_state.pt', map_location='cpu', weights_only=True), strict=True)
        m.torch.manual_seed(config['ddpm_seed'])
        models['ddpm_ema_state'] = m.UNet_conditional(c_in=2, c_out=2, time_dim=256, num_class1=2, num_class2=2)
        models['ddpm_ema_state'].load_state_dict(m.torch.load(SOURCE / 'ddpm_ema_state.pt', map_location='cpu', weights_only=True), strict=True)
        models['ddpm_ema_state'].eval().requires_grad_(False)
        for name, model in models.items():
            before[name] = m.state_hash(model)
            if before[name] != checkpoint_meta[name]['tensor_state_sha256']:
                raise ValueError('loaded tensor digest differs: ' + name)
            for key, tensor in model.state_dict().items():
                m.finite(tensor, 'loaded ' + name + ':' + key)
        dump(out / 'loaded_state_identity.json', {'before': before, 'expected': contract['checkpoint_tensor_digests']})
        state.update(setup_seconds=time.perf_counter()-process_started, phase='generation')
        dump(out / 'status.json', state)
        schedule = m.noise_schedule(config['noise_steps'])
        generation = m.generate(config, models['vae_state'], models['ddpm_ema_state'], schedule, out, rows)
        state['generation'] = generation
        state['generation_cap_exceeded'] = generation['elapsed_seconds'] > 1800
        if state['generation_cap_exceeded']:
            raise TimeoutError('new generation cap exceeded; no retry or extension')
        state['phase'] = 'frozen_reconstruction_export'
        # Restore serialized statistics only: deliberately no constructor/fit call.
        scaler = m.FeatureScaler.__new__(m.FeatureScaler)
        scaler.stats = read_json(SOURCE / 'train_feature_scaler.json')
        if scaler.stats['fit_scope'] != 'train only' or scaler.stats['fit_record_ids'] != list(range(279)):
            raise ValueError('saved scaler provenance differs')
        dump(out / 'restored_scaler_identity.json', {'source': str(SOURCE / 'train_feature_scaler.json'),
            'sha256': sha(SOURCE / 'train_feature_scaler.json'), 'fit_called': False})
        state['source_train_read_attempted'] = True
        dump(out / 'status.json', state)
        train = m.read_records(config['train'], config['train_sha256'], config['train_rows'], config['train_cohort'], config['train_cohort_sha256'], config['train_idx_start'])
        state['source_train_read_validated'] = True
        state['source_dev_read_attempted'] = True
        dump(out / 'status.json', state)
        dev = m.read_records(config['dev'], config['dev_sha256'], config['dev_rows'], config['dev_cohort'], config['dev_cohort_sha256'], config['dev_idx_start'])
        state['source_dev_read_validated'] = True
        if {r.protein for r in train} & {r.protein for r in dev}:
            raise ValueError('train/dev overlap')
        export_started = time.perf_counter()
        m.reconstruction_export(config, models['vae_state'], train, scaler, out, 'train')
        m.reconstruction_export(config, models['vae_state'], dev, scaler, out, 'dev')
        state['export_seconds'] = time.perf_counter()-export_started
        state['state'] = 'recovery_completed' if generation['successful_execution'] == 32 else 'recovery_completed_with_failure'
    except (Exception, RecoveryInterrupt) as exc:
        state.update(state='recovery_failed', error=repr(exc), traceback=traceback.format_exc())
        print(state['traceback'], file=sys.stderr, flush=True)
    finally:
        # Allow best-effort records after a deadline/signal. The external bounded
        # supervisor still owns termination; any overrun is explicitly a failure.
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)
        finalization_errors = []
        identity = {'before': before, 'expected': contract['checkpoint_tensor_digests'], 'after': {}}
        for name, model in models.items():
            try:
                identity['after'][name] = m.state_hash(model)
            except Exception as exc:
                finalization_errors.append({'phase': 'model_identity', 'name': name, 'error': repr(exc)})
        identity['both_loaded_verified_and_unchanged'] = len(before) == 2 and identity['after'] == before and before == contract['checkpoint_tensor_digests']
        if not identity['both_loaded_verified_and_unchanged']:
            state['state'] = 'recovery_failed'
        try:
            original_errors = verify_original_snapshot(contract)
            core_errors = verify_bindings(contract['core_bindings'])
            state['original_inputs_unchanged'] = not original_errors and not core_errors
            dump(out / 'input_integrity_after.json', {'original_errors': original_errors, 'core_errors': core_errors})
            if original_errors or core_errors:
                state['state'] = 'recovery_failed'
        except Exception as exc:
            finalization_errors.append({'phase': 'input_identity', 'error': repr(exc)})
            state['original_inputs_unchanged'] = False
        state['model_states_unchanged'] = identity['both_loaded_verified_and_unchanged']
        state['process_seconds_before_final_status_io'] = time.perf_counter()-process_started
        state['outer_cap_exceeded'] = state['process_seconds_before_final_status_io'] > 1980
        if state['outer_cap_exceeded'] or finalization_errors:
            state['state'] = 'recovery_failed'
        state['finalization_errors'] = finalization_errors
        state['phase'] = 'closed'
        dump(out / 'model_identity_after.json', identity)
        dump(out / 'status.json', state)
        dump(out / 'manifest.json', {'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json']})
    return 0 if state['state'] == 'recovery_completed' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('contract', 'plan', 'receipt', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    return execute(a.contract, a.plan, a.receipt, a.output)


if __name__ == '__main__':
    raise SystemExit(main())
