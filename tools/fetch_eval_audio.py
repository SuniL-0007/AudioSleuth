"""Fetch the real-speech evaluation excerpt; the audio itself is never committed.

Source: NASA Johnson Space Center, "Houston We Have a Podcast", episode 424,
"AI at NASA". NASA media is generally not subject to U.S. copyright and may be
used for educational purposes with NASA credited as the source and no implied
endorsement (https://www.nasa.gov/nasa-brand-center/images-and-media/).
The excerpt skips the show's intro music and starts at the interview.
"""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
URL = "https://traffic.megaphone.fm/NATIONALAERONAUTICSANDSPACEADMINISTRATION8144252345.mp3"
SOURCE_SHA256 = "b01e757dc9e3500c8827b05bd592c1f9d8a975946c0d84dc6907155bdd668b41"
START_SECONDS, END_SECONDS = 140, 860
OUTPUT = ROOT / "evaluation" / "data" / "nasa_ai_at_nasa.wav"


def main() -> int:
    """Download, verify the exact source bytes, then write a 16 kHz mono excerpt."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audiosleuth-fetch-") as temporary:
        source = Path(temporary) / "episode.mp3"
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != SOURCE_SHA256:
            print(
                "The downloaded episode differs from the judged one (the host may have "
                f"inserted ads or re-encoded it).\nExpected {SOURCE_SHA256}\nGot      {digest}",
                file=sys.stderr,
            )
            return 1
        with sf.SoundFile(source) as stream:
            rate = stream.samplerate
            stream.seek(START_SECONDS * rate)
            data = stream.read(
                (END_SECONDS - START_SECONDS) * rate, dtype="float32", always_2d=True
            )
    factor = math.gcd(rate, 16_000)
    mono = resample_poly(data.mean(axis=1), 16_000 // factor, rate // factor)
    sf.write(OUTPUT, np.clip(mono, -1, 1).astype(np.float32), 16_000, subtype="PCM_16")
    print(f"Wrote {OUTPUT} ({len(mono) / 16_000:.1f} s)")
    print(f"Excerpt SHA-256 {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
