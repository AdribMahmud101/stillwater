"""Private Modal Glimmer service; deployment and GPU spend are not automatic.

Uses the spike's pinned llama.cpp / GGUF setup. No public endpoint is exposed.
Caller must supply privacy-processed input and enforce a budget; the model and
GPU service must be verified on real inference before production use.
"""
import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

app = modal.App('stillwater-glimmer')
MODEL_FILES = ('Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf', 'mmproj-Muse-Glimmer-30B-Q4_K_M.gguf')
weights = modal.Volume.from_name('stillwater-glimmer-spike-weights', create_if_missing=False)
def _bake_weights():
    """Copy the GGUF files from the network Volume into the image layer once, at build time.

    Reading 17 GB from a Volume at every cold start took about 10 minutes; image layers
    are cached on the worker and load much faster.
    """
    import os
    import shutil
    os.makedirs('/models', exist_ok=True)
    for name in MODEL_FILES:
        shutil.copyfile('/weights/' + name, '/models/' + name)


image = (
    modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.12')
    .apt_install('git', 'cmake', 'build-essential', 'libcurl4-openssl-dev')
    .run_commands(
        'git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp',
        'cd /opt/llama.cpp && git checkout f8def7fe168bab245fbf15d3f18b26dbb1ef73c8',
        'cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--allow-shlib-undefined',
        'cmake --build /opt/llama.cpp/build --target llama-server -j4',
    )
    .run_function(_bake_weights, volumes={'/weights': weights})
)


def _validate_payload(payload):
    if not isinstance(payload, dict) or payload.get('model') != 'muse-glimmer':
        raise ValueError('invalid model')
    tokens = payload.get('max_tokens')
    if type(tokens) is not int or not 1 <= tokens <= 2048:
        raise ValueError('max_tokens must be between 1 and 2048')
    if not isinstance(payload.get('messages'), list) or not payload['messages']:
        raise ValueError('messages required')
    if payload.get('stream'):
        raise ValueError('streaming is not supported')
    if len(json.dumps(payload).encode()) > 16 * 1024 * 1024:
        raise ValueError('request too large')


def _request_glimmer(payload, opener):
    _validate_payload(payload)
    request = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        with opener(request, timeout=240) as response:
            result = json.load(response)
        if not isinstance(result, dict) or not isinstance(result.get('choices'), list) or not result['choices']:
            raise ValueError('invalid completion')
        return result
    except (OSError, ValueError, TypeError):
        raise RuntimeError('Glimmer inference failed') from None


@app.cls(image=image, gpu='L4', volumes={'/weights': weights}, timeout=900,
         max_containers=1, scaledown_window=600)
class GlimmerService:
    """Private RPC class; local llama-server is never exposed as a web endpoint."""

    @modal.enter()
    def start(self):
        self.proc = subprocess.Popen([
            '/opt/llama.cpp/build/bin/llama-server',
            '-m', '/models/' + MODEL_FILES[0],
            '--mmproj', '/models/' + MODEL_FILES[1],
            '-a', 'muse-glimmer', '-ngl', '99', '-c', '4096', '-np', '1',
            '--host', '127.0.0.1', '--port', '8080',
            '--chat-template-kwargs', '{"reasoning_strength":"low"}',
        ])  # stdout/stderr go to Modal logs so startup failures are visible
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError('Glimmer server exited during startup')
            try:
                with self.opener.open('http://127.0.0.1:8080/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        self.proc.terminate()
        raise TimeoutError('Glimmer server startup timed out')

    @modal.method()
    def ping(self):
        """Cheap call used to pre-warm the container before the first real request."""
        return 'ok'

    @modal.method()
    def chat(self, payload: dict):
        started = time.monotonic()
        response = _request_glimmer(payload, self.opener.open)
        # This measures call duration, not billed GPU time or a complete cost.
        return {'response': response, 'inference_seconds': time.monotonic() - started}

    @modal.exit()
    def stop(self):
        if getattr(self, 'proc', None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
