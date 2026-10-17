"""Sealed evaluator: Ternary-Bonsai-2-9B on the GTX 1080 (sm_61), production setup (PQ2_0 + MTP). Run from the repo root
after the build (research/build.py).

Quality gate (repeat 1 only): llama-perplexity KL divergence of the PQ2_0 model against the baseline's logits, at three
micro-batch sizes so every kernel path is covered: 512 (prefill), 3 (the MTP verify step), 1 (single-token decode).
Each: mean KLD <= 1e-4 and same top token >= 99.5%. The baseline (CHATROOM_KIND=baseline) writes the references.
Speed (every repeat): llama-server with the production flags (PQ2_0-MTP, draft-mtp n-max 2, KV f16, flash attention,
1 slot), a ~4k-token shared context then 3 greedy answers of 384 tokens (decode at depth ~4k, prefix reused).
Metric: tps_mtp = generated tokens / generation seconds (higher is better) in $CHATROOM_RESULT.
Guard: prefill tokens/s of the context >= 97% of the slowest baseline repeat.
References live OUTSIDE the worktrees (every worker's worktree reads the same baseline).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
BIN = ROOT / 'build-win' / 'bin'
REF = Path(os.environ.get('BONSAI_REF_DIR', 'research-ref'))  # baseline references, outside the worktrees
MODELS = Path(os.environ.get('BONSAI_MODELS_DIR', 'models'))
PQ2 = MODELS / 'bonsai2-9b-Ternary-Bonsai-2-9B-PQ2_0.gguf'
MTP = MODELS / 'bonsai2-9b-Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf'
TEXT = ROOT / 'research' / 'kld.txt'
PORT = 8099
KLD_MAX, TOP1_MIN, GUARD = 1e-4, 99.5, 0.97
UBATCHES = (512, 3, 1)
QUESTIONS = ['Riassumi in italiano, in modo dettagliato, il testo qui sopra.',
             'Write a Python function that parses the results table described above and computes the averages.',
             'Explain step by step, in English, how the experiments above were measured and what could be improved.']


def run(cmd, timeout):
    print('$', ' '.join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors='replace')
    out = (r.stdout or '') + (r.stderr or '')
    if r.returncode:
        print(out[-4000:])
        sys.exit(f'FAILED ({r.returncode}): {cmd[0]}')
    return out


def quality(baseline):
    for ub in UBATCHES:
        ref = REF / f'ref-ub{ub}.kld'
        cmd = [str(BIN / 'llama-perplexity.exe'), '-m', str(PQ2), '-f', str(TEXT), '-c', '512', '--chunks', '4',
               '-ngl', '999', '-fa', 'on', '-b', '512', '-ub', str(ub), '--kl-divergence-base', str(ref)]
        if baseline:
            run(cmd, 1200)
            print(f'baseline logits written (ubatch {ub}): {ref}')
            continue
        if not ref.exists():
            sys.exit(f'no baseline reference {ref}: the baseline run writes it')
        out = run(cmd + ['--kl-divergence'], 1200)
        kld = re.search(r'Mean\s+KLD:\s*([-0-9.e+]+)', out)
        top = re.search(r'Same top p:\s*([0-9.]+)', out)
        if not (kld and top):
            print(out[-3000:])
            sys.exit(f'quality gate (ubatch {ub}): could not read the KLD / top token lines')
        kld, top = float(kld.group(1)), float(top.group(1))
        print(f'quality ubatch {ub}: mean KLD {kld:.3e} (max {KLD_MAX:.0e}), same top token {top:.3f}% (min {TOP1_MIN}%)')
        if kld > KLD_MAX or top < TOP1_MIN:
            sys.exit(f'quality gate FAILED at ubatch {ub}: the change alters the model output')


def post(path, body, timeout=600):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def speed():
    log = open(ROOT / 'build-win' / 'research-server.log', 'w', encoding='utf-8')
    srv = subprocess.Popen([str(BIN / 'llama-server.exe'), '--host', '127.0.0.1', '--port', str(PORT), '-m', str(MTP),
                            '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '--spec-draft-n-min', '0',
                            '--spec-draft-p-min', '0', '-c', '16384', '--parallel', '1', '-ngl', '999', '-fa', 'on',
                            '--jinja', '--reasoning', 'off', '--cache-ram', '0'], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    sys.exit('llama-server died at startup, see build-win/research-server.log')
                time.sleep(1)
        context = TEXT.read_text(encoding='utf-8')[:16000]
        base = {'temperature': 0.0, 'top_k': 1, 'seed': 1, 'cache_prompt': True}
        first = post('/v1/chat/completions', {**base, 'max_tokens': 1,
                                              'messages': [{'role': 'user', 'content': context + '\n\nOK?'}]})
        pps = first['timings']['prompt_per_second']
        gen_tok = gen_ms = 0.0
        acc = drafted = 0
        for q in QUESTIONS:
            r = post('/v1/chat/completions', {**base, 'max_tokens': 384,
                                              'messages': [{'role': 'user', 'content': context + '\n\n' + q}]})
            t = r['timings']
            gen_tok += t['predicted_n']
            gen_ms += t['predicted_ms']
            drafted += t.get('draft_n', 0) or 0
            acc += t.get('draft_n_accepted', 0) or 0
        tps = gen_tok / (gen_ms / 1000.0)
        print(f'speed: prefill {pps:.1f} tok/s ({first["timings"]["prompt_n"]} tok), decode {tps:.2f} tok/s over '
              f'{int(gen_tok)} tok, draft accepted {acc}/{drafted}')
        return tps, pps
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except Exception:
            srv.kill()
        log.close()


def main():
    print('protocol: PQ2_0-MTP draft-mtp n-max 2, ctx 16384, KV f16, fa on, 1 slot; context 16000 chars + 3x384 greedy; '
          'gate KLD ub 512/3/1 <= 1e-4, top >= 99.5%; prefill guard 97%')
    baseline = os.environ.get('CHATROOM_KIND') == 'baseline'
    repeat = int(os.environ.get('CHATROOM_REPEAT', '1'))
    REF.mkdir(parents=True, exist_ok=True)
    if repeat == 1:
        quality(baseline)
    tps, pps = speed()
    if baseline:
        (REF / f'baseline-r{repeat}.json').write_text(json.dumps({'tps_mtp': tps, 'prefill': pps}), encoding='utf-8')
    else:
        refs = [json.loads(p.read_text(encoding='utf-8'))['prefill'] for p in REF.glob('baseline-r*.json')]
        if not refs:
            sys.exit('no baseline speed reference: the baseline run writes it')
        if pps < GUARD * min(refs):
            sys.exit(f'speed guard FAILED: prefill {pps:.1f} < {GUARD:.0%} of baseline {min(refs):.1f}')
    line = f'tps_mtp: {tps:.3f}'
    res = os.environ.get('CHATROOM_RESULT')
    if res:
        Path(res).write_text(line + '\n', encoding='utf-8')
    print('result', line.replace(':', ' ='))


if __name__ == '__main__':
    main()
