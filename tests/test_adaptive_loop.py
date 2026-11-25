#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive looped core: per-token exit and inter-pass messages (docs/LOOP.md).

Checks, on the CPU binary of this machine (TARGET=macos on Apple Silicon, TARGET=aarch64 on Linux):

  1. builds: TARGET=macos (or the native CPU target) and TARGET=aarch64, which composes serve.om
     (its compile is the serve.om syntax check on macOS; the ELF is not run there);
  2. preservation: a binary built from the pre-adaptive model.om/cli.om/serve.om and the new binary print the
     same eval, sample, kvcmp and train results and write the same checkpoint bytes;
  3. thresh 0, gain 0: adapteval prints eval's numbers bit for bit (the telemetry hooks run);
  4. thresh > 0: rows leave the loop at different passes (heterogeneous exit depths) and the
     loss changes;
  5. gain > 0: inter-pass messages are deterministic (two runs, same bytes) and change the loss;
  6. adaptcmp: the cached decode (ofwdi) and the full forward (ofwd) give 0 differing logits and
     the same per-row exit passes, message norms and pass-0 deltas, also across the context limit
     and on a build whose activations cross into the windowed bank layout;
  7. teeth: mutant x (the cached decode skips the end-of-pass hook) fails adaptcmp with exit 43;
  8. adaptsamp: deterministic, and with 0 0 its greedy tokens are kvcmp's;
  9. usage refusals.

Usage: python3 tests/test_adaptive_loop.py [--keep]   (OMC=/path/to/omc to pick the compiler)
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OWNED = ("native/model.om", "native/cli.om", "native/serve.om")

FAILS = []
PASSES = []


def check(cond, name, detail=""):
    if cond:
        PASSES.append(name)
        print("PASS " + name)
    else:
        FAILS.append(name)
        print("FAIL " + name + (": " + detail if detail else ""))


def run(args, cwd=None, env=None, ok=(0,)):
    p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode("latin-1")
    if ok is not None and p.returncode not in ok:
        raise RuntimeError("%s -> %d\n%s\n%s" % (" ".join(args), p.returncode, out,
                                                 p.stderr.decode("latin-1")))
    return p.returncode, out


def build(target, out, src_root=REPO):
    env = dict(os.environ, TARGET=target, OUT=out)
    run(["sh", os.path.join(src_root, "scripts", "build.sh")], env=env)


def cpu_target():
    if platform.system() == "Darwin":
        return "macos", "ollm-macos"
    return "aarch64", "ollm"


def compile_om(om_path, binary):
    """Compile an already composed unit (used for mutant / small-pool builds)."""
    omc = os.environ.get("OMC") or shutil.which("omc") or os.path.expanduser("~/.local/bin/omc")
    with open(om_path, "rb") as f, open(binary + ".tmp", "wb") as g:
        p = subprocess.run([omc], stdin=f, stdout=g, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("omc %s -> %d %s" % (om_path, p.returncode, p.stderr.decode()))
    os.chmod(binary + ".tmp", 0o755)
    os.replace(binary + ".tmp", binary)
    if platform.system() == "Darwin" and shutil.which("codesign"):
        subprocess.run(["codesign", "-s", "-", "-f", binary], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def variant(src_om, dst_om, subs):
    text = open(src_om).read()
    for old, new in subs:
        if text.count(old) != 1:
            raise RuntimeError("variant: %r occurs %d times" % (old, text.count(old)))
        text = text.replace(old, new)
    with open(dst_om, "w") as f:
        f.write(text)


def line(out, tag):
    for ln in out.splitlines():
        if ln.startswith(tag + " "):
            return ln.split()[1:]
    raise RuntimeError("no %s line in:\n%s" % (tag, out))


def lines(out, tag):
    return [ln.split()[1:] for ln in out.splitlines() if ln.startswith(tag + " ")]


def strip_ns(out, tags):
    """kvcmp lines without their timing fields (the last field of KI/KF, the last two of KVCMP)."""
    res = []
    for ln in out.splitlines():
        f = ln.split()
        if not f:
            continue
        if f[0] in ("KI", "KF"):
            res.append(" ".join(f[:-1]))
        elif f[0] == "KVCMP":
            res.append(" ".join(f[:3]))
        elif f[0] in tags:
            res.append(ln)
    return res


def base_ref():
    """The commit before the adaptive loop: the parent of the oldest commit that added cmdadcmp
    (HEAD while it is uncommitted); ADAPT_BASE_REF overrides."""
    if os.environ.get("ADAPT_BASE_REF"):
        return os.environ["ADAPT_BASE_REF"]
    _, out = run(["git", "-C", REPO, "log", "--format=%H", "-S", "fn cmdadcmp", "--", "native/cli.om"])
    shas = out.split()
    return shas[-1] + "^" if shas else "HEAD"


def baseline_tree(work):
    """A copy of the scripts and native tree with the base versions of the three owned files."""
    root = os.path.join(work, "head")
    shutil.copytree(os.path.join(REPO, "native"), os.path.join(root, "native"))
    shutil.copytree(os.path.join(REPO, "scripts"), os.path.join(root, "scripts"))
    ref = base_ref()
    print("baseline: %s of %s" % (ref, ", ".join(OWNED)))
    for rel in OWNED:
        _, txt = run(["git", "-C", REPO, "show", ref + ":" + rel])
        with open(os.path.join(root, rel), "w", encoding="latin-1") as f:
            f.write(txt)
    if "fn cmdadcmp" in open(os.path.join(root, "native/cli.om"), encoding="latin-1").read():
        raise RuntimeError("baseline already has the adaptive loop")
    return root


def main():
    keep = "--keep" in sys.argv
    work = tempfile.mkdtemp(prefix="adaptive-loop-")
    tgt, binname = cpu_target()
    try:
        # ---- 1. builds
        bdir = os.path.join(work, "new")
        build(tgt, bdir)
        B = os.path.join(bdir, binname)
        check(os.path.exists(B), "build TARGET=%s" % tgt)
        sdir = os.path.join(work, "serve")
        build("aarch64", sdir)
        elf = open(os.path.join(sdir, "ollm"), "rb").read(4)
        som = open(os.path.join(sdir, "ollm.om")).read()
        check(elf == b"\x7fELF" and "svnl_emit_loop" in som and "fn cmdadcmp" in som,
              "build TARGET=aarch64 (serve.om with adaptive-loop fields compiles)")
        for rel in OWNED:
            n = sum(1 for _ in open(os.path.join(REPO, rel)))
            check(n < 8192, "line count %s = %d" % (rel, n))

        # ---- data, tokenizer, checkpoints
        text = os.path.join(work, "text.txt")
        with open(text, "wb") as f:
            for rel in ("docs/LOOP.md", "README.md"):
                f.write(open(os.path.join(REPO, rel), "rb").read())
        M = os.path.join(work, "m.bin")
        TK = os.path.join(work, "toks.bin")
        run([B, "bpefit", text, M, "300"])
        run([B, "bpeenc", M, "300", text, TK])
        LP = os.path.join(work, "lp.ck")
        run([B, "init", LP, "300", "32", "32", "2", "64", "4", "2", "7", "100", "3000", "1", "1", "2", "3"])
        run([B, "train", LP, TK, "20", "20"])
        FL = os.path.join(work, "flat.ck")
        run([B, "init", FL, "300", "32", "32", "2", "64", "4", "2", "7", "100", "3000"])
        run([B, "train", FL, TK, "10", "10"])
        prompt = "The loop"

        # ---- 2. preservation against HEAD's three files
        head = baseline_tree(work)
        hdir = os.path.join(work, "headbin")
        build(tgt, hdir, src_root=head)
        H = os.path.join(hdir, binname)
        for ck in (LP, FL):
            tag = os.path.basename(ck)
            for args in (["eval", ck, TK, "8", M],
                         ["sample", ck, "1", "24", "0", prompt, M],
                         ["sample", ck, "5", "24", "700", prompt, M],
                         ["nsample", ck, "3", "12", "0", prompt, M, "2", "1500"]):
                a = run([H] + args)[1]
                b = run([B] + args)[1]
                check(a == b, "preserve %s %s" % (" ".join(os.path.basename(x) for x in args[:1] + args[2:4]), tag),
                      "%r != %r" % (a[:200], b[:200]))
            a = strip_ns(run([H, "kvcmp", ck, "40", prompt, M])[1], ())
            b = strip_ns(run([B, "kvcmp", ck, "40", prompt, M])[1], ())
            check(a == b and a[-1].endswith(" 0"), "preserve kvcmp %s (0 diff)" % tag)
            c1 = os.path.join(work, "h-" + tag)
            c2 = os.path.join(work, "n-" + tag)
            shutil.copy(ck, c1)
            shutil.copy(ck, c2)
            a = run([H, "train", c1, TK, "25", "5"])[1]
            b = run([B, "train", c2, TK, "25", "5"])[1]
            check(a == b and open(c1, "rb").read() == open(c2, "rb").read(),
                  "preserve train %s (same lines, same checkpoint bytes)" % tag)

        # ---- 3. thresh 0 gain 0 == eval
        for ck in (LP, FL):
            e = line(run([B, "eval", ck, TK, "8", M])[1], "E")
            out = run([B, "adapteval", ck, TK, "8", "0", "0", M])[1]
            check(line(out, "AE") == e, "adapteval 0 0 == eval bits (%s)" % os.path.basename(ck))
        out0 = run([B, "adapteval", LP, TK, "8", "0", "0", M])[1]
        ax0 = line(out0, "AX")
        e = line(out0, "AE")
        np_, rows, exited = int(ax0[0]), int(ax0[1]), int(ax0[2])
        d0min, d0max = int(ax0[4]), int(ax0[5])
        check(np_ == 3 and rows == 8 * 32 and exited == 0 and line(out0, "AH") == ["0", "0", str(rows)],
              "adapteval 0 0: every row runs all NP passes", " ".join(ax0))
        check(0 < d0min < d0max, "pass-0 deltas measured (%d .. %d micro)" % (d0min, d0max))

        # ---- 4. heterogeneous exit depths
        th = (d0min + d0max) // 2
        out = run([B, "adapteval", LP, TK, "8", str(th), "0", M])[1]
        ah = [int(x) for x in line(out, "AH")]
        ax = line(out, "AX")
        check(sum(1 for c in ah if c > 0) >= 2 and sum(ah) == rows and ah[0] > 0,
              "thresh %d: heterogeneous per-token exit depths AH %s" % (th, ah))
        check(int(ax[2]) > 0 and line(out, "AE") != e, "thresh > 0 changes the loss (%s vs %s)"
              % (line(out, "AE")[2], e[2]))
        # an exit threshold above every delta stops every row after pass 1
        out = run([B, "adapteval", LP, TK, "8", str(d0max + 1), "0", M])[1]
        check(line(out, "AH") == [str(rows), "0", "0"], "thresh above every delta: all rows exit after pass 1")

        # ---- 5. inter-pass messages
        G = "300000"
        m1 = run([B, "adapteval", LP, TK, "8", "0", G, M])[1]
        m2 = run([B, "adapteval", LP, TK, "8", "0", G, M])[1]
        axm = line(m1, "AX")
        check(m1 == m2, "messages deterministic (two runs byte-identical)")
        check(int(axm[6]) > 0 and axm[7] != ax0[7] and line(m1, "AE") != e,
              "messages flow (norm %s milli) and change the loss" % axm[6])
        check(line(m1, "AH") == ["0", "0", str(rows)] and axm[4:6] == ax0[4:6],
              "messages leave pass 0 untouched and no row exits at thresh 0")
        both = run([B, "adapteval", LP, TK, "8", str(th), G, M])[1]
        check(both == run([B, "adapteval", LP, TK, "8", str(th), G, M])[1], "exit+messages deterministic")

        # ---- 6. cached decode == full forward
        for t_, g_ in (("0", "0"), (str(th), "0"), ("0", G), (str(th), G)):
            rc, out = run([B, "adaptcmp", LP, "40", prompt, M, t_, g_], ok=None)
            f = line(out, "ADAPTCMP")
            check(rc == 0 and f[:3] == ["40", "0", "0"], "adaptcmp thresh %s gain %s: 0 logit diff, 0 telemetry diff" % (t_, g_), out[-300:])
        rc, out = run([B, "adaptcmp", LP, "40", prompt, M, str(th), G], ok=None)
        xps = set(r[4] for r in lines(out, "AI"))
        check(len(xps) >= 2 and max(int(r[1]) for r in lines(out, "AI")) == 32,
              "adaptcmp crosses the context limit with mixed exit passes %s" % sorted(xps))

        # ---- 6b. windowed activation banks (shrunk OAC) give the same bits
        wom = os.path.join(work, "win.om")
        variant(os.path.join(bdir, binname + ".om"), wom,
                [("const OAC ulong 16777216", "const OAC ulong 80000"),
                 ("global oa f64 16777216", "global oa f64 80000")])
        W = os.path.join(work, "ollm-win")
        compile_om(wom, W)
        a = run([W, "adapteval", LP, TK, "8", str(th), G, M])[1]
        check(a == both, "windowed bank layout: adapteval bits equal")
        rc, out = run([W, "adaptcmp", LP, "40", prompt, M, str(th), G], ok=None)
        check(rc == 0 and line(out, "ADAPTCMP")[:3] == ["40", "0", "0"], "windowed bank layout: adaptcmp 0 diff")

        # ---- 7. mutant x
        mom = os.path.join(work, "mutx.om")
        variant(os.path.join(bdir, binname + ".om"), mom, [("const OMUTAX ulong 1", "const OMUTAX ulong 0")])
        X = os.path.join(work, "ollm-mutx")
        compile_om(mom, X)
        rc, out = run([X, "adaptcmp", LP, "40", prompt, M, str(th), G], ok=None)
        f = line(out, "ADAPTCMP")
        check(rc == 43 and (int(f[1]) > 0 or int(f[2]) > 0), "mutant x caught by adaptcmp (exit %d, %s)" % (rc, f[:3]))
        rc, out = run([X, "adaptcmp", LP, "40", prompt, M, "0", "0"], ok=None)
        f = line(out, "ADAPTCMP")
        check(rc == 43 and f[1] == "0" and int(f[2]) > 0,
              "mutant x at thresh 0 gain 0: logits unchanged, missing telemetry caught (%s)" % f[:3])

        # ---- 8. adaptsamp
        s1 = run([B, "adaptsamp", LP, "9", "20", "800", prompt, M, str(th), G])[1]
        s2 = run([B, "adaptsamp", LP, "9", "20", "800", prompt, M, str(th), G])[1]
        check(s1 == s2 and len(lines(s1, "AS")) >= 1 and line(s1, "ADAPTSAMP"), "adaptsamp deterministic")
        sg = run([B, "adaptsamp", LP, "1", "20", "0", prompt, M, "0", "0"])[1]
        kv = run([B, "kvcmp", LP, "20", prompt, M])[1]
        ki = [r[3] for r in lines(kv, "KI")][:10]
        asg = [r[2] for r in lines(sg, "AS")][:10]
        check(ki == asg, "adaptsamp 0 0 greedy tokens == kvcmp's", "%s vs %s" % (ki, asg))

        # ---- 9. refusals
        check(run([B, "adapteval", LP, TK, "8", "0"], ok=None)[0] == 2, "adapteval usage exit 2")
        check(run([B, "adaptcmp", LP, "40", prompt, M, "0"], ok=None)[0] == 2, "adaptcmp usage exit 2")
        B256 = os.path.join(work, "b256.ck")
        run([B, "init", B256, "256", "16", "16", "2", "32", "2", "2", "7", "100", "3000", "1", "0", "2", "2"])
        check(run([B, "adaptcmp", B256, "4", prompt, M, "0", "0"], ok=None)[0] == 2, "adaptcmp V = 256 exit 2")
    finally:
        if keep:
            print("work dir kept: " + work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("%s adaptive-loop %d passed %d failed" % ("PASS" if not FAILS else "FAIL", len(PASSES), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
