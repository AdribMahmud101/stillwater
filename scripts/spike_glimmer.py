"""Unverified one-shot Modal GPU feasibility spike for Muse Glimmer vision.

No deployment happens by importing this file. After verifying Modal auth, GPU
billing, and a spending allowance, run explicitly:

    uv run modal run scripts/spike_glimmer.py --image-path path/to/crop.jpg --confirm-gpu-spend

The first run builds an image and downloads roughly 19 GB into a Modal Volume.
This spike is not the production witness or a production budget guard.
References: https://dev.meta.ai/docs/muse-glimmer/deploy
            https://modal.com/docs/guide/gpu
"""
import json
import time
from pathlib import Path

import modal

app = modal.App('stillwater-glimmer-spike')
weights = modal.Volume.from_name('stillwater-glimmer-spike-weights', create_if_missing=True)
image = (
    modal.Image.from_registry('nvidia/cuda:12.4.1-devel-ubuntu22.04', add_python='3.12')
    .apt_install('git', 'cmake', 'build-essential', 'libcurl4-openssl-dev')
    .pip_install('huggingface_hub==0.35.3')
    .run_commands(
        'git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp',
        'cd /opt/llama.cpp && git checkout f8def7fe168bab245fbf15d3f18b26dbb1ef73c8',
        'cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--allow-shlib-undefined',
        'cmake --build /opt/llama.cpp/build --target llama-server -j4',
    )
)


@app.function(image=image, volumes={'/weights': weights}, timeout=3600, max_containers=1)
def prepare_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id='meta-models/Muse-Glimmer-30B-GGUF', local_dir='/weights',
        allow_patterns=['Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf', 'mmproj-Muse-Glimmer-30B-Q4_K_M.gguf'],
    )
    weights.commit()
    return 'Weights downloaded.'


@app.function(image=image, gpu='L4', volumes={'/weights': weights}, timeout=900, max_containers=1, scaledown_window=2)
def run_crop(image_bytes: bytes, content_type: str):
    import base64
    import subprocess
    import urllib.error
    import urllib.request

    weights.reload()
    if content_type not in ('image/png', 'image/jpeg') or not 0 < len(image_bytes) <= 10 * 1024 * 1024:
        raise ValueError('Use a PNG or JPEG crop at most 10 MiB.')
    started = time.monotonic()
    # Server listens only on loopback inside this short-lived container.
    proc = subprocess.Popen([
        '/opt/llama.cpp/build/bin/llama-server',
        '-m', '/weights/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf',
        '--mmproj', '/weights/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf',
        '-a', 'muse-glimmer', '-ngl', '99', '-c', '4096', '-np', '1',
        '--host', '127.0.0.1', '--port', '8080',
        '--chat-template-kwargs', '{"reasoning_strength":"low"}',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('Model server exited during startup.')
            try:
                with opener.open('http://127.0.0.1:8080/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        else:
            raise TimeoutError('Model server failed to become ready.')
        ready_seconds = time.monotonic() - started
        image_url = f'data:{content_type};base64,' + base64.b64encode(image_bytes).decode('ascii')
        payload = {
            'model': 'muse-glimmer', 'max_tokens': 512,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe this container as JSON with keys holds_water, covered, water_visible, orientation, exposed_to_rain. Use yes, no, or unsure except orientation: upright, inverted, on_side, or unsure. Use unsure when not visible. Return JSON only.'},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ]}],
        }
        request = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        inference_started = time.monotonic()
        with opener.open(request, timeout=240) as response:
            result = json.load(response)
        content = result['choices'][0]['message'].get('content')
        try:
            parsed = json.loads(content)
            valid_json = isinstance(parsed, dict)
        except (ValueError, TypeError):
            valid_json = False
        return {'response': result, 'valid_json_object': valid_json, 'cold_start_seconds': ready_seconds,
                'inference_seconds': time.monotonic() - inference_started,
                'elapsed_gpu_seconds_lower_bound': time.monotonic() - started}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@app.local_entrypoint()
def main(image_path: str, confirm_gpu_spend: bool = False):
    if not confirm_gpu_spend:
        raise SystemExit('Check Modal account and budget, then pass --confirm-gpu-spend to run the spike.')
    path = Path(image_path)
    image_bytes = path.read_bytes()
    ext = path.suffix.lower()
    content_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext)
    if not content_type or not 0 < len(image_bytes) <= 10 * 1024 * 1024:
        raise SystemExit('Provide a nonempty PNG or JPEG crop of at most 10 MiB.')
    prepare_weights.remote()
    result = run_crop.remote(image_bytes, content_type)
    out = Path('tests/fixtures/glimmer_spike.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved response to {out}; valid JSON object: {result["valid_json_object"]}; '
          f'GPU execution lower bound: {result["elapsed_gpu_seconds_lower_bound"]:.1f} s.')
