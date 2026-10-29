"""Cancellable native trials with explicit preparation/IPC/cleanup accounting.

A deadline cancels child work, including an unresponsive Rust call. It cannot
preempt a parent np.save, np.load, process creation or OS cleanup. Hence this is
not a hard real-time or total-call deadline; parent overrun is reported.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
import numpy as np
from .planner import PlanningError


class TrialTimeout(PlanningError):
    pass


class TrialWorker:
    def __init__(self, name, codes, weights, threads, scratch_bytes, deadline):
        self.name = name
        self._process = None
        self._thread = None
        self._temp = tempfile.TemporaryDirectory(prefix='econhdfe-probe-')
        self._root = Path(self._temp.name)
        self._responses = queue.Queue()
        self._config = dict(backend=name, threads=int(threads), scratch_bytes=int(scratch_bytes))
        self._closed = False
        try:
            self._check(deadline)
            np.save(self._root/'codes.npy', codes, allow_pickle=False)
            np.save(self._root/'weights.npy', weights, allow_pickle=False)
            self._check(deadline)
            env = os.environ.copy()
            # Use this installed/source package in the subprocess, not another version.
            package_root = str(Path(__file__).resolve().parents[1])
            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            env['OMP_NUM_THREADS'] = str(threads)
            env['RAYON_NUM_THREADS'] = str(threads)
            self._stderr = open(self._root/'stderr.log', 'w', encoding='utf-8')
            self._process = subprocess.Popen(
                [sys.executable, '-u', '-m', 'econhdfe_planner._bounded_worker', str(self._root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding='utf-8', bufsize=1, env=env,
            )
            self._thread = threading.Thread(target=self._read, args=(self._process.stdout, self._responses), daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(deadline):
        if deadline is not None and perf_counter() >= deadline:
            raise TrialTimeout('optional native-work deadline exhausted')

    @staticmethod
    def _read(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    def _request(self, request, deadline):
        self._check(deadline)
        if self._closed or self._process is None:
            raise PlanningError('worker closed')
        try:
            self._process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
            self._process.stdin.flush()
            remaining = None if deadline is None else max(0.0, deadline-perf_counter())
            answer = self._responses.get(timeout=remaining)
        except queue.Empty as exc:
            self.close()
            raise TrialTimeout('native trial interrupted at wall deadline') from exc
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise PlanningError('native worker pipe failed') from exc
        if answer is None:
            detail = (self._root/'stderr.log').read_text(encoding='utf-8')[-1000:]
            self.close()
            raise PlanningError('native worker exited: '+detail)
        try:
            response = json.loads(answer)
        except (ValueError, TypeError) as exc:
            self.close()
            raise PlanningError('invalid native worker response') from exc
        if not response.get('ok'):
            raise PlanningError(response.get('error', 'native trial rejected'))
        return response

    def project(self, data, tol, max_iter, *, deadline=None, refresh=False):
        self._check(deadline)
        np.save(self._root/'input.npy', data, allow_pickle=False)
        response = self._request(dict(self._config, op='project', tol=float(tol),
                                      max_iter=int(max_iter), refresh=bool(refresh)), deadline)
        self._check(deadline)
        result = np.load(self._root/'output.npy', allow_pickle=False)
        self._check(deadline)
        return result, dict(response['info'], worker_timing=response.get('timing', {}))

    def update_weights(self, weights):
        if self._closed:
            raise PlanningError('worker closed')
        np.save(self._root/'weights.npy', weights, allow_pickle=False)
        self._request(dict(op='update'), None)

    @property
    def alive(self):
        return self._process is not None and self._process.poll() is None

    def close(self):
        if self._closed:
            return
        self._closed = True
        p = self._process
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=2.0)
            if p.stdin:
                p.stdin.close()
            if self._thread is not None:
                self._thread.join(timeout=0.2)
            if p.stdout:
                p.stdout.close()
        if hasattr(self, '_stderr'):
            self._stderr.close()
        self._temp.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
