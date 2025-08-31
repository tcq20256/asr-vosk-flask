import io
import os
import wave
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from vosk import Model, KaldiRecognizer

# ========== 基础配置 ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "vosk-model-small-cn-0.22"
# MODEL_PATH = BASE_DIR / "models" / "vosk-model-cn-0.22"
SAMPLE_RATE = 16000
MAX_FILE_MB = 25                 # 上传体积上限（与前端提示保持一致）
FRAMES_PER_CHUNK = 4000          # 识别时每次读取的“帧数”，4000 帧≈0.25s
FFMPEG_BIN = "ffmpeg"            # 如需自定义路径，可改为绝对路径

app = Flask(__name__, static_folder="web", static_url_path="/")
CORS(app)

# Flask 层面的硬性限制（早于视图函数执行）
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# ========== 模型加载 ==========
if not MODEL_PATH.exists():
    raise RuntimeError(f"未找到模型目录: {MODEL_PATH}")
model = Model(model_path=str(MODEL_PATH))

def ffmpeg_available() -> bool:
    """检测 ffmpeg 是否可用"""
    return shutil.which(FFMPEG_BIN) is not None

# ========== 工具：用 ffmpeg 转成 16k 单声道 wav ==========
def to_wav_mono16k(src_bytes: bytes) -> bytes:
    if not ffmpeg_available():
        raise RuntimeError("未检测到 ffmpeg，请先安装（conda-forge 或 yum 源）")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_in:
        f_in.write(src_bytes)
        in_path = f_in.name

    out_path = in_path + ".wav"
    try:
        # -ac 1: 单声道  -ar 16000: 16k  -f wav: 输出 wav
        cmd = [
            FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "wav", out_path
        ]
        subprocess.run(cmd, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except Exception:
                pass

# ========== 识别核心 ==========
def recognize_wav_bytes(wav_bytes: bytes) -> dict:
    """
    返回 {"text": str, "seconds": float}
    """
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            # 基本校验（ffmpeg 已确保正确格式，这里双保险）
            if wf.getnchannels() != 1:
                raise ValueError("期望单声道 wav")
            if wf.getsampwidth() != 2:
                raise ValueError("期望 16-bit PCM")
            if wf.getframerate() != SAMPLE_RATE:
                raise ValueError(f"期望采样率 {SAMPLE_RATE}")

            rec = KaldiRecognizer(model, SAMPLE_RATE)
            rec.SetWords(True)

            result_text = []
            while True:
                data = wf.readframes(FRAMES_PER_CHUNK)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    partial = json.loads(rec.Result()).get("text", "")
                    if partial:
                        result_text.append(partial)

            final = json.loads(rec.FinalResult()).get("text", "")
            if final:
                result_text.append(final)

            # 简单清理
            text = " ".join([t.strip() for t in result_text if t.strip()])
            text = " ".join(text.split())

            seconds = wf.getnframes() / float(wf.getframerate())
            return {"text": text, "seconds": seconds}

# ========== 路由 ==========
@app.route("/")
def index():
    # 提供一个简易前端（可选）
    return app.send_static_file("index.html")

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ffmpeg": ffmpeg_available(),
        "model_path": str(MODEL_PATH),
        "sample_rate": SAMPLE_RATE,
        "max_file_mb": MAX_FILE_MB
    })

@app.post("/api/stt")
def stt():
    # 统计对象
    proc = psutil.Process(os.getpid())
    num_cpus = psutil.cpu_count(logical=True) or 1

    try:
        if "audio" not in request.files:
            return jsonify({"error": "缺少文件字段 'audio'"}), 400

        f = request.files["audio"]

        # 原始字节长度（读一遍到内存）
        f.seek(0, io.SEEK_END)
        file_bytes = f.tell()
        f.seek(0, io.SEEK_SET)

        if file_bytes > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify({"error": f"文件过大，限制 {MAX_FILE_MB}MB"}), 400

        raw = f.read()

        # ========== 计时起点 ==========
        wall0 = time.perf_counter()
        cpu0 = proc.cpu_times()
        rss0 = proc.memory_info().rss

        # ========== 转码 ==========
        dec_t0 = time.perf_counter()
        try:
            wav_bytes = to_wav_mono16k(raw)
        except subprocess.CalledProcessError:
            return jsonify({"error": "音频解码失败，请确认格式/编解码是否可被 ffmpeg 处理"}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        dec_t1 = time.perf_counter()

        # ========== 识别 ==========
        rec_t0 = time.perf_counter()
        recog = recognize_wav_bytes(wav_bytes)  # {"text":..., "seconds":...}
        rec_t1 = time.perf_counter()

        # ========== 统计收尾 ==========
        cpu1 = proc.cpu_times()
        rss1 = proc.memory_info().rss
        wall1 = time.perf_counter()

        wall_ms = int((wall1 - wall0) * 1000)
        decode_ms = int((dec_t1 - dec_t0) * 1000)
        recog_ms = int((rec_t1 - rec_t0) * 1000)
        cpu_time_ms = int(((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) * 1000)

        # CPU 占比（按核数归一）
        cpu_percent_est = 0.0
        if wall_ms > 0:
            cpu_percent_est = min(100.0, max(0.0, (cpu_time_ms / wall_ms) * 100.0 / num_cpus))

        rss_mb = rss1 / (1024 * 1024)
        rss_delta_mb = (rss1 - rss0) / (1024 * 1024)

        return jsonify({
            "text": recog.get("text", ""),
            "audio_seconds": round(recog.get("seconds", 0.0), 2),

            # 时延指标
            "latency_ms": wall_ms,           # 端到端（接口内）
            "decode_ms": decode_ms,          # ffmpeg 转码
            "recognize_ms": recog_ms,        # vosk 识别

            # 资源占用
            "cpu_time_ms": cpu_time_ms,
            "cpu_percent_est": round(cpu_percent_est, 1),
            "rss_mb": round(rss_mb, 1),
            "rss_delta_mb": round(rss_delta_mb, 1),

            # 输入信息
            "file_bytes": file_bytes,
            "limit_mb": MAX_FILE_MB
        })

    except wave.Error as e:
        return jsonify({"error": f"WAV 解析失败：{e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 低配机：不开 debug，单进程
    app.run(host="0.0.0.0", port=5000, threaded=False)
