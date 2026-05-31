import base64, os, tempfile, runpod, torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

print("Chargement faster-whisper...")
_whisper = WhisperModel(MODEL_SIZE,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8")

print("Chargement pyannote...")
_diarize = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
if torch.cuda.is_available():
    _diarize = _diarize.to(torch.device("cuda"))
print("Modeles charges — pret.")

def _decode_audio(b64, tmp_dir):
    path = os.path.join(tmp_dir, "audio.mp3")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path

def handler(job):
    inp      = job.get("input", {})
    b64      = inp.get("audio", "")
    language = inp.get("language", "fr")
    diarize  = inp.get("diarize", True)
    if not b64:
        return {"error": "Champ audio manquant"}
    with tempfile.TemporaryDirectory() as tmp:
        path = _decode_audio(b64, tmp)
        try:
            if diarize:
                dz = _diarize(path)
                spk_segs = [{"speaker": s, "start": t.start, "end": t.end}
                            for t, _, s in dz.itertracks(yield_label=True)]
                wh_segs, info = _whisper.transcribe(path, language=language)
                wh_list = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in wh_segs]
                def dom(ws, ss):
                    ov = {}
                    for sp in ss:
                        lo = max(ws["start"], sp["start"])
                        hi = min(ws["end"], sp["end"])
                        if hi > lo:
                            ov[sp["speaker"]] = ov.get(sp["speaker"], 0) + (hi - lo)
                    return max(ov, key=ov.get) if ov else "SPEAKER_00"
                merged = []
                for ws in wh_list:
                    spk = dom(ws, spk_segs)
                    if merged and merged[-1]["speaker"] == spk:
                        merged[-1]["text"] += " " + ws["text"]
                        merged[-1]["end"]   = ws["end"]
                    else:
                        merged.append({"speaker": spk,
                                       "start": round(ws["start"], 2),
                                       "end":   round(ws["end"], 2),
                                       "text":  ws["text"]})
                full = "\n".join(
                    f'[{s["speaker"]} — {int(s["start"]//60):02d}:{int(s["start"]%60):02d}] {s["text"]}'
                    for s in merged)
                speakers = sorted({s["speaker"] for s in merged})
                return {"segments": merged, "full_text": full,
                        "speakers_count": len(speakers), "language": info.language}
            else:
                segs, info = _whisper.transcribe(path, language=language)
                return {"segments": [], "full_text": " ".join(s.text.strip() for s in segs),
                        "speakers_count": 1, "language": info.language}
        except Exception as e:
            return {"error": str(e), "full_text": "", "segments": []}

runpod.serverless.start({"handler": handler})
