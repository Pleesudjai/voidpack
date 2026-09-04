"""Detect any modification to the independent test suite.

The executor model is forbidden from editing tests/. That instruction was violated
once on 2026-09-03, so it is now enforced by checksum rather than by request.

    python tools_check_tests.py --freeze   record the current checksums
    python tools_check_tests.py            verify nothing changed, exit 1 if it did

Run the verify form before every commit and before the submission.

Checksums are taken over content with CRLF normalised to LF since 2026-09-04, so the
same suite verifies on a Windows checkout, a Linux checkout and a Dropbox copy alike.
"""
import glob
import hashlib
import os
import sys

MANIFEST = "tests/CHECKSUMS.txt"


def digests():
    out = {}
    for p in sorted(glob.glob("tests/test_*.py")):
        # Hash with line endings normalised, so a checkout that rewrites CRLF
        # to LF, or a Dropbox sync that does the reverse, is not reported as an
        # edit. Content is what the contract protects, not the newline bytes.
        raw = open(p, "rb").read().replace(b"\r\n", b"\n")
        h = hashlib.sha256(raw).hexdigest()
        out[p.replace("\\", "/")] = h
    return out


def freeze():
    d = digests()
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("# Independent test suite checksums. Written by tools_check_tests.py.\n")
        f.write("# The executor model must never change these files. Regenerate only\n")
        f.write("# when a human deliberately edits a test.\n")
        for k, v in d.items():
            f.write("%s  %s\n" % (v, k))
    print("froze %d test files" % len(d))


def verify():
    if not os.path.exists(MANIFEST):
        print("NO MANIFEST. Run --freeze first."); return 1
    want = {}
    for line in open(MANIFEST, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        h, p = line.split(None, 1)
        want[p.strip()] = h
    have = digests()
    bad = []
    for p, h in want.items():
        if p not in have:
            bad.append("DELETED  " + p)
        elif have[p] != h:
            bad.append("MODIFIED " + p)
    for p in have:
        if p not in want:
            bad.append("ADDED    " + p)
    if bad:
        print("TEST SUITE INTEGRITY FAILURE")
        for b in bad:
            print("   " + b)
        print("\nThe tests are the contract. If the executor rewrote them, restore them")
        print("before trusting any gate that passed.")
        return 1
    print("test suite unmodified, %d files" % len(have))
    return 0


if __name__ == "__main__":
    sys.exit(freeze() or 0 if "--freeze" in sys.argv else verify())
