"""
Textphonic — turns text into music with spoken narration layered over it.
Built with Streamlit so it deploys free on Streamlit Community Cloud.
"""

import io
import tempfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from gtts import gTTS
from pydub import AudioSegment

SR = 44100

SCALES = {
    "Major pentatonic": [0, 2, 4, 7, 9],
    "Minor pentatonic": [0, 3, 5, 7, 10],
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Natural minor": [0, 2, 3, 5, 7, 8, 10],
    "Chromatic": list(range(12)),
}

ROOTS = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}

VOWELS = set("aeiou")


# ---------- text -> note sequence ----------

def build_score(text, root_name, scale_name, base_octave, spread, bpm):
    root = ROOTS[root_name]
    scale = SCALES[scale_name]
    beat = 60.0 / bpm

    events = []
    t = 0.0

    for raw in text:
        ch = raw.lower()

        if ch == " ":
            t += beat * 0.25
            continue
        if ch == ",":
            t += beat * 0.6
            continue
        if ch in ".;":
            t += beat * 1.1
            continue
        if not (ch.isalnum() and ch.isascii()):
            if events:
                if ch == "!":
                    events[-1]["vel"] = min(1.0, events[-1]["vel"] + 0.35)
                if ch == "?":
                    events[-1]["bend"] = 4
            t += beat * 0.3
            continue

        if ch.isdigit():
            degree_source = int(ch)
        else:
            degree_source = ord(ch) - ord("a")

        degree = degree_source % len(scale)
        octave_shift = (degree_source // len(scale)) % spread
        semis = scale[degree] + 12 * octave_shift
        is_caps = raw != ch and raw.isalpha()
        midi = (base_octave + (1 if is_caps else 0)) * 12 + root + semis + 12

        is_vowel = ch in VOWELS
        dur = beat * 0.55 if is_vowel else beat * 0.3
        vel = 0.95 if is_caps else 0.7

        events.append({"midi": midi, "dur": dur, "vel": vel, "time": t, "bend": 0})
        t += dur * 0.9

    total_time = t + beat
    return events, total_time


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------- instruments ----------

def adsr(n, sr, attack=0.01, decay=0.1, sustain=0.6, release=0.15):
    env = np.full(n, sustain, dtype=np.float64)
    a = min(int(sr * attack), n)
    d = min(int(sr * decay), max(n - a, 0))
    r = min(int(sr * release), n)

    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)
    if r > 0:
        start_val = env[n - r - 1] if n - r - 1 >= 0 else sustain
        env[n - r:] = np.linspace(start_val, 0, r)
    return env


def synth_soft(freq, dur, vel, sr=SR):
    n = max(1, int(sr * dur * 1.2))
    t = np.arange(n) / sr
    wave = np.sin(2 * np.pi * freq * t)
    env = adsr(n, sr, 0.02, 0.15, 0.3, 0.4)
    return wave * env * vel


def synth_fm(freq, dur, vel, sr=SR):
    n = max(1, int(sr * dur * 1.3))
    t = np.arange(n) / sr
    mod = np.sin(2 * np.pi * freq * 2 * t) * 6
    wave = np.sin(2 * np.pi * freq * t + mod)
    env = adsr(n, sr, 0.01, 0.3, 0.1, 0.5)
    return wave * env * vel


def synth_membrane(freq, dur, vel, sr=SR):
    n = max(1, int(sr * max(dur, 0.25)))
    t = np.arange(n) / sr
    pitch_env = freq * (1 + 2 * np.exp(-t * 30))
    phase = 2 * np.pi * np.cumsum(pitch_env) / sr
    wave = np.sin(phase)
    env = np.exp(-t * 10)
    return wave * env * vel


def synth_pluck(freq, dur, vel, sr=SR):
    # Karplus-Strong plucked string
    n = max(1, int(sr * max(dur, 0.3)))
    period = max(2, int(sr / freq))
    buf = np.random.uniform(-1, 1, period)
    out = np.zeros(n)
    for i in range(n):
        idx = i % period
        nxt = (i + 1) % period
        out[i] = buf[idx]
        buf[idx] = 0.5 * (buf[idx] + buf[nxt]) * 0.996
    env = np.exp(-np.arange(n) / sr * 3)
    return out * env * vel


INSTRUMENTS = {
    "Plucked string": synth_pluck,
    "Soft synth": synth_soft,
    "FM bell": synth_fm,
    "Membrane / mallet": synth_membrane,
}


def render_instrumental(events, total_time, instrument, sr=SR):
    n_total = int(sr * (total_time + 1))
    master = np.zeros(n_total)
    synth_fn = INSTRUMENTS[instrument]

    for ev in events:
        freq = midi_to_freq(ev["midi"] + ev.get("bend", 0))
        wave = synth_fn(freq, ev["dur"], ev["vel"], sr)
        start = int(ev["time"] * sr)
        end = start + len(wave)
        if end > len(master):
            master = np.pad(master, (0, end - len(master)))
        master[start:end] += wave

    peak = np.max(np.abs(master)) if len(master) else 0
    if peak > 0:
        master = master / peak * 0.9
    return master


def numpy_to_segment(arr, sr=SR):
    arr16 = np.int16(np.clip(arr, -1, 1) * 32767)
    return AudioSegment(arr16.tobytes(), frame_rate=sr, sample_width=2, channels=1)


# ---------- vocals ----------

def synthesize_narration(text):
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return AudioSegment.from_file(buf, format="mp3")


def mix(instrumental_seg, vocal_seg, music_gain_db=-9, vocal_gain_db=-1):
    total_len = max(len(instrumental_seg), len(vocal_seg))

    instr = instrumental_seg + music_gain_db
    if len(instr) < total_len:
        instr += AudioSegment.silent(duration=total_len - len(instr))

    vocal = vocal_seg + vocal_gain_db
    if len(vocal) < total_len:
        vocal += AudioSegment.silent(duration=total_len - len(vocal))

    return instr.overlay(vocal)


# ---------- visualization ----------

def make_piano_roll(events, total_time):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1c1916")
    ax.set_facecolor("#171412")

    for ev in events:
        ax.barh(ev["midi"], ev["dur"] * 0.9, left=ev["time"], height=0.8, color="#c89b3c")

    if total_time > 0:
        ax.set_xlim(0, total_time)
    ax.set_xlabel("seconds", color="#a99f8f")
    ax.set_ylabel("pitch (MIDI)", color="#a99f8f")
    ax.tick_params(colors="#a99f8f")
    for spine in ax.spines.values():
        spine.set_color("#3a352d")
    fig.tight_layout()
    return fig


# ---------- generation ----------

def generate(text, root, scale, instrument, bpm, base_octave, spread):
    events, total_time = build_score(text, root, scale, int(base_octave), int(spread), int(bpm))
    if not events:
        return None, None, "That text has no letters or numbers to turn into notes."

    instrumental = render_instrumental(events, total_time, instrument)
    instr_seg = numpy_to_segment(instrumental)

    vocal_seg = synthesize_narration(text)

    mixed = mix(instr_seg, vocal_seg)

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    mixed.export(out_path, format="wav")

    fig = make_piano_roll(events, total_time)
    return out_path, fig, None


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Textphonic", page_icon="🎼")

st.title("Textphonic")
st.write(
    "Type something, pick a key and an instrument, and hear it played back "
    "as music with your words spoken over the top."
)

text_in = st.text_area(
    "Your text",
    value="to be, or not to be, that is the question",
    height=100,
)

col1, col2, col3 = st.columns(3)
with col1:
    root_in = st.selectbox("Root note", list(ROOTS.keys()), index=list(ROOTS.keys()).index("G"))
with col2:
    scale_in = st.selectbox("Scale", list(SCALES.keys()), index=0)
with col3:
    instrument_in = st.selectbox("Instrument", list(INSTRUMENTS.keys()), index=0)

col4, col5, col6 = st.columns(3)
with col4:
    bpm_in = st.slider("Tempo (bpm)", 60, 240, 150)
with col5:
    octave_in = st.slider("Base octave", 2, 6, 4)
with col6:
    spread_in = st.slider("Octave range", 1, 3, 2)

if st.button("Generate", type="primary"):
    if not text_in.strip():
        st.error("Type some text first.")
    else:
        with st.spinner("Composing and narrating..."):
            out_path, fig, error = generate(
                text_in, root_in, scale_in, instrument_in, bpm_in, octave_in, spread_in
            )
        if error:
            st.error(error)
        else:
            with open(out_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button("Download WAV", audio_bytes, file_name="textphonic.wav")
            st.pyplot(fig)
