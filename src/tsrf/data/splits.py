import hashlib
import re
from collections import defaultdict

#: `video-<id>-frame-<n>-<hash>.<ext>`; the id itself may contain dashes.
FRAME_RE = re.compile(r"^video-(?P<video>.+?)-frame-(?P<frame>\d+)-(?P<hash>[^.]+)")


def parse_frame_name(name):
    stem = name.replace("\\", "/").rsplit("/", 1)[-1]
    m = FRAME_RE.match(stem)
    if not m:
        raise ValueError(f"filename does not match FLIR ADAS convention: {name!r}")
    return m.group("video"), int(m.group("frame"))


def group_by_video(names):
    groups = defaultdict(list)
    for n in names:
        video, _ = parse_frame_name(n)
        groups[video].append(n)
    for v in groups.values():
        v.sort(key=lambda n: parse_frame_name(n)[1])
    return dict(groups)


def _stable_unit(video_id, seed):
    h = hashlib.blake2b(f"{seed}:{video_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def split_by_video(names, fractions=(0.8, 0.1, 0.1), seed=1337):
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    total = sum(fractions)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    groups = group_by_video(names)
    train_cut = fractions[0]
    val_cut = fractions[0] + fractions[1]

    out = {"train": [], "val": [], "test": []}
    for video, frames in groups.items():
        u = _stable_unit(video, seed)
        split = "train" if u < train_cut else ("val" if u < val_cut else "test")
        out[split].extend(frames)

    for k in out:
        out[k].sort()
    assert_no_video_leakage(out)
    return out


def assert_no_video_leakage(splits):
    seen = {}
    for split, names in splits.items():
        for video in {parse_frame_name(n)[0] for n in names}:
            if video in seen and seen[video] != split:
                raise AssertionError(
                    f"video {video!r} appears in both {seen[video]!r} and {split!r}")
            seen[video] = split
    return True


def split_summary(splits):
    lines = []
    for split, names in splits.items():
        videos = {parse_frame_name(n)[0] for n in names}
        lines.append(f"{split:5s}: {len(names):6d} frames from {len(videos):4d} videos")
    return "\n".join(lines)
