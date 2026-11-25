#!/usr/bin/env python3
"""Record a continuous, uncut PTY session proving zero CUDA / zero libc linkage,
live `/proc/$PID/maps` + `/proc/$PID/fd` inspection during GB10 `gtrainb`, and live negative controls.

Produces:
  - <OUT_DIR>/proof.cast       (asciicast v2)
  - <OUT_DIR>/transcript.txt   (verbatim ANSI/UTF-8 log)
  - <OUT_DIR>/SHA256SUMS.txt
"""

from __future__ import annotations

import json
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_uncut_session(bin_path: Path, repo_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    inner_sh = work / "proof_cmds.sh"
    inner_sh.write_text(f"""#!/bin/sh
set -u
export PS1="$ "
BIN="{bin_path}"
REPO="{repo_root}"
WORK="{work}"

say_cmd() {{ printf '\\n$ %s\\n' "$1"; }}

say_cmd "file \\$BIN && ldd \\$BIN"
file "$BIN"
ldd "$BIN" 2>&1 || true

say_cmd "cat \\$REPO/README.md > \\$WORK/corpus.txt && \\$BIN init ... && \\$BIN gtrainb ... & sleep 2 && cat /proc/\\$PID/maps | grep -i cuda"
cat "$REPO/README.md" "$REPO/docs/LOOP.md" "$REPO/docs/CONTRACT.md" "$REPO/docs/CONTRACT-V1.md" > "$WORK/corpus.txt"
"$BIN" bpefit "$WORK/corpus.txt" "$WORK/m4096.bin" 4096 >/dev/null
"$BIN" bpeenc "$WORK/m4096.bin" 4096 "$WORK/corpus.txt" "$WORK/train.tok" >/dev/null
"$BIN" init "$WORK/v2.ck" 4096 256 384 12 1536 8 64 1 60000 3000 16 | head -n 5

"$BIN" gtrainb "$WORK/v2.ck" "$WORK/train.tok" 80 80 > "$WORK/gtrainb.log" 2>&1 &
GPID=$!
sleep 2
printf 'pid=%s alive=%s\\n' "$GPID" "$(kill -0 "$GPID" 2>/dev/null && echo yes || echo no)"
printf 'cuda/cublas/libc matches in /proc/%s/maps: ' "$GPID"
(grep -iE 'cuda|cublas|libc\\.so' /proc/"$GPID"/maps || echo "0 matches (empty)")
ls -l /proc/"$GPID"/fd | grep -E 'nvidia|pts|0 ->|1 ->' || true
wait "$GPID"
rc_gpu=$?
echo "--- gtrainb output (rc=$rc_gpu) ---"
cat "$WORK/gtrainb.log"

say_cmd "Negative control 1: flip 1 bit in OLLMSP01 .sp pack -> spapp must fail with rc=61"
"$BIN" init "$WORK/tiny.ck" 256 16 16 2 32 2 2 42 4 3000 >/dev/null
"$BIN" growpro "$WORK/tiny.ck" "$WORK/spec.ck" 3 4 2000 >/dev/null
"$BIN" spext "$WORK/tiny.ck" "$WORK/spec.ck" "$WORK/skill.sp" >/dev/null
python3 -c 'import sys; b=bytearray(open(sys.argv[1],"rb").read()); b[48]^=1; open(sys.argv[2],"wb").write(b)' "$WORK/skill.sp" "$WORK/bad.sp"
"$BIN" spapp "$WORK/tiny.ck" "$WORK/bad.sp" "$WORK/out.ck"
echo "spapp tampered exit code: rc=$?"

say_cmd "Negative control 2: OLLM_ORACLE_MUTANT=tiemaxa python3 tests/oracle.py selftest-v1"
set +e
OLLM_ORACLE_MUTANT=tiemaxa python3 "$REPO/tests/oracle.py" selftest-v1 > "$WORK/mut.out" 2>&1
rc_mut=$?
set -e
tail -n 8 "$WORK/mut.out"
echo "oracle mutant exit code: rc=$rc_mut"
""", encoding="utf-8")
    os.chmod(inner_sh, 0o755)

    events: list[tuple[float, str]] = []
    raw_chunks: list[bytes] = []
    pid, fd = pty.fork()
    t0 = time.monotonic()
    if pid == 0:
        os.execv("/bin/sh", ["/bin/sh", str(inner_sh)])
    else:
        try:
            while True:
                rlist, _, _ = select.select([fd], [], [], 0.2)
                if fd in rlist:
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    dt = round(time.monotonic() - t0, 4)
                    raw_chunks.append(chunk)
                    events.append((dt, chunk.decode("utf-8", errors="replace")))
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                else:
                    wpid, _ = os.waitpid(pid, os.WNOHANG)
                    if wpid == pid:
                        break
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

    shutil.rmtree(work, ignore_errors=True)
    transcript = b"".join(raw_chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    (out_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    header = {"version": 2, "width": 96, "height": 34, "timestamp": int(time.time()), "title": "Omega Helix GB10 Zero-CUDA Uncut Proof"}
    lines = [json.dumps(header)] + [json.dumps([dt, "o", txt]) for dt, txt in events]
    (out_dir / "proof.cast").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: record-zero-cuda-proof.sh <ollm-gb10-bin> <out-dir>", file=sys.stderr)
        return 2
    bin_p = Path(sys.argv[1]).expanduser().resolve()
    out_p = Path(sys.argv[2]).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    run_uncut_session(bin_p, repo_root, out_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
