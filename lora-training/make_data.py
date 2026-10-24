"""make_data.py: builds a train.jsonl for train.py from AgentNet (Windows steps) and Multimodal-Mind2Web (web steps),
in the format TernaryPilot v1 was trained on. Coordinates are integers on a 0-1000 grid over the image.

  python make_data.py --agentnet agentnet_win_mac_18k.jsonl --agentnet-images DIR \
                      --mind2web "train-*.parquet" --out data/train.jsonl --images out_images/ [--exclude "bench/*.png"]

AgentNet (MIT): completed tasks with alignment_score >= 8, Windows only (no macOS words in the first observations), one
example per step with the history of previous actions as text; plus, on part of the steps, a screen reading example
(the step's observation) and a click example (the step's target). --agentnet-images is a folder with the extracted PNGs.
Mind2Web (OpenRAIL): the full-page screenshot is cropped to a 1280x800 window around the target; one action example per
step plus a click example on the element name. --exclude: images whose perceptual hash is close to any of these files
are skipped (use it to keep your evaluation screenshots out of training)."""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import random
import re

from PIL import Image

P_CLIC = ('Localize an element on the GUI image according to the provided target and output a click position.\n'
          ' * You must output a valid JSON following the format: {"x": int, "y": int}, with x and y normalized to 0-1000\n'
          ' Your target is:\n')
AZIONI = ("click(x, y), double_click(x, y), right_click(x, y), move(x, y), drag(x, y), write(text), press(key), "
          "hotkey(keys), scroll(amount), wait(), done()")
P_LETTURA = "Describe the current screen: the application or website, its state, and the important visible text."


def p_azione(dove, compito, storia):
    st = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(storia[-8:])) or "none"
    return (f"You are operating a {dove}.\nTask: {compito}\nPrevious actions:\n{st}\n"
            f"Look at the current screen and give the next action. Coordinates are integers normalized to 0-1000.\n"
            f"Available actions: {AZIONI}\nReply as:\nThought: <short reasoning>\nAction: <one action>")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:16]


def dhash(im, n=16):
    g = im.convert("L").resize((n + 1, n))
    px = list(g.getdata())
    return sum(1 << i for i in range(n * n) if px[(i // n) * (n + 1) + i % n] > px[(i // n) * (n + 1) + i % n + 1])


def frase(t, n):  # cut at a sentence end within n characters
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= n:
        return t
    c = t[:n]
    k = max(c.rfind(". "), c.rfind(".\n"))
    return c[:k + 1] if k > n // 2 else c.rstrip() + "..."


def converti(code):
    """pyautogui / computer.* code with 0-1 coordinates -> actions on the 0-1000 grid; None if not recognised"""
    out = []
    for riga in code.strip().splitlines():
        riga = riga.strip()
        if not riga:
            continue
        try:
            nodo = ast.parse(riga).body[0].value
            nome = nodo.func.attr
            kw = {k.arg: ast.literal_eval(k.value) for k in nodo.keywords}
            args = [ast.literal_eval(a) for a in nodo.args]
        except Exception:
            return None

        def xy():
            x = kw.get("x", args[0] if args else 0)
            y = kw.get("y", args[1] if len(args) > 1 else 0)
            return "x=%d, y=%d" % (round(x * 1000), round(y * 1000))

        if nome == "click":
            n, b = kw.get("clicks", 1), kw.get("button", "left")
            out.append(("right_click" if b == "right" else "double_click" if n == 2 else "click") + f"({xy()})")
        elif nome in ("doubleClick", "rightClick", "moveTo", "dragTo", "tripleClick"):
            out.append({"doubleClick": "double_click", "rightClick": "right_click", "moveTo": "move", "dragTo": "drag",
                        "tripleClick": "triple_click"}[nome] + f"({xy()})")
        elif nome in ("write", "typewrite"):
            out.append("write(%s)" % json.dumps(kw.get("message", args[0] if args else ""), ensure_ascii=False))
        elif nome == "press":
            k = kw.get("keys", args[0] if args else "")
            out.append("press(%s)" % json.dumps("+".join(k) if isinstance(k, list) else k))
        elif nome == "hotkey":
            k = args[0] if len(args) == 1 else args if args else kw.get("keys", [])
            k = k if isinstance(k, list) else [k]
            if not all(isinstance(x, str) for x in k):
                return None
            out.append("hotkey(%s)" % json.dumps("+".join(k)))
        elif nome in ("scroll", "hscroll"):
            out.append("scroll(%d)" % int(kw.get("clicks", args[0] if args else 0)))
        elif nome == "terminate":
            out.append("done()")
        elif nome == "wait":
            out.append("wait()")
        else:
            return None
    return "\n".join(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agentnet"), ap.add_argument("--agentnet-images"), ap.add_argument("--mind2web")
    ap.add_argument("--out", required=True), ap.add_argument("--images", required=True), ap.add_argument("--exclude")
    ap.add_argument("--max-agentnet", type=int, default=900), ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(a.images, exist_ok=True)
    vietate = [dhash(Image.open(f)) for f in glob.glob(a.exclude, recursive=True)] if a.exclude else []
    vietata = lambda im: any(bin(dhash(im) ^ v).count("1") <= 12 for v in vietate)  # noqa: E731
    blocchi = []

    if a.agentnet:
        mac = re.compile(r"\bmac ?os\b|\bfinder\b|\bdock\b|\bmenu bar\b|\bcommand key|\bcmd\b|safari|spotlight", re.I)
        win = re.compile(r"\bwindows\b|taskbar|start menu|file explorer|\bwps\b|ctrl\+", re.I)
        passi = []
        for riga in open(a.agentnet, encoding="utf-8"):
            d = json.loads(riga)
            if not d.get("task_completed") or (d.get("alignment_score") or 0) < 8:
                continue
            oss = " ".join(s["value"].get("observation", "") for s in d["traj"][:3])
            if mac.search(oss) or not win.search(oss):
                continue
            compito, storia = d.get("natural_language_task") or d["instruction"], []
            for s in d["traj"]:
                v, img = s["value"], os.path.join(a.agentnet_images, s["image"])
                act = converti(v.get("code") or "")
                if act and os.path.exists(img) and v.get("last_step_correct", True) is not False:
                    passi.append((img, compito, list(storia), v, act))
                storia.append(frase(v.get("action", ""), 160))
        rnd.shuffle(passi)
        for img, compito, storia, v, act in passi[:a.max_agentnet]:
            if vietata(Image.open(img)):
                continue
            k = md5(img)
            b = [{"cache": k, "img": img, "user": p_azione("Windows computer", compito, storia),
                  "risposta": f"Thought: {frase(v.get('action', ''), 220)}\nAction: {act}", "tipo": "action_pc"}]
            if rnd.random() < 0.35 and v.get("observation"):
                b.append({"cache": k, "img": img, "user": P_LETTURA, "risposta": frase(v["observation"], 900), "tipo": "reading"})
            mm = re.fullmatch(r"(?:double_|right_)?click\((x=\d+, y=\d+)\)", act)
            t = re.sub(r"^(double[- ]|right[- ])?click (on |the )*", "", frase(v.get("action", ""), 120), flags=re.I).rstrip(".")
            if mm and 3 <= len(t) <= 100:
                x, y = re.findall(r"\d+", mm.group(1))
                b.append({"cache": k, "img": img, "user": P_CLIC + t, "risposta": '{"x": %s, "y": %s}' % (x, y), "tipo": "click_pc"})
            blocchi.append(b)

    if a.mind2web:
        import pyarrow.parquet as pq
        for f in sorted(glob.glob(a.mind2web)):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
                for r in batch.to_pylist():
                    try:
                        pos = json.loads(r["pos_candidates"][0])
                        x0, y0, w, h = [float(v) for v in json.loads(pos["attributes"])["bounding_box_rect"].split(",")]
                        im = Image.open(io.BytesIO(r["screenshot"]["bytes"])).convert("RGB")
                    except Exception:
                        continue
                    W, H = im.size
                    if w <= 0 or h <= 0 or W < 800:
                        continue
                    top = int(min(max(0, y0 + h / 2 - 400), max(0, H - 800)))
                    cr = im.crop((0, top, W, min(H, top + 800)))
                    cx, cy = (x0 + w / 2) / W, (y0 + h / 2 - top) / cr.size[1]
                    if not (0 < cx < 1 and 0 < cy < 1) or vietata(cr):
                        continue
                    out = os.path.join(a.images, f"{r['action_uid']}.png")
                    cr.save(out)
                    k = md5(out)
                    px, py = round(cx * 1000), round(cy * 1000)
                    opr = json.loads(r["operation"]) if isinstance(r["operation"], str) else r["operation"]
                    op, val = opr["op"], opr.get("value", "")
                    act = f"click(x={px}, y={py})" + (f"\nwrite({json.dumps(val, ensure_ascii=False)})" if op == "TYPE" else
                                                     f"\nselect({json.dumps(val, ensure_ascii=False)})" if op == "SELECT" else "")
                    elem = re.sub(r"^\[\w+\]\s*", "", r["target_action_reprs"].split(" -> ")[0]).strip()
                    pensiero = f"The next step is to {op.lower()} the {elem or 'target element'}" + (f" with \"{val}\"" if val else "") + "."
                    storia = [s.replace(" -> ", ": ") for s in r["action_reprs"][:int(r["target_action_index"])]]
                    b = [{"cache": k, "img": out, "user": p_azione("web browser", r["confirmed_task"], storia),
                          "risposta": f"Thought: {pensiero}\nAction: {act}", "tipo": "action_web"}]
                    if 5 <= len(elem) <= 80:
                        b.append({"cache": k, "img": out, "user": P_CLIC + elem, "risposta": '{"x": %d, "y": %d}' % (px, py), "tipo": "click_web"})
                    blocchi.append(b)

    # order: blocks of the same screenshot stay inside one window of 4 (shared prefix), types are mixed
    rnd.shuffle(blocchi)
    finestre, aperte = [], []
    for b in blocchi:
        for f in aperte:
            if sum(map(len, f)) + len(b) <= 4:
                f.append(b)
                break
        else:
            aperte.append([b])
        for f in [f for f in aperte if sum(map(len, f)) == 4]:
            finestre.append(f)
            aperte.remove(f)
    finestre += aperte
    rnd.shuffle(finestre)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for f in finestre:
            for b in f:
                for e in b:
                    fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("examples", sum(len(b) for b in blocchi), "screenshots", len(blocchi))


if __name__ == "__main__":
    main()
