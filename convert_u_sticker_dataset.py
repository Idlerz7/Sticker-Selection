import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


SPEAKER_RE = re.compile(r"^\[speaker(\d+)\]")


@dataclass
class ConvertStats:
    total_samples: int = 0
    kept_samples: int = 0
    dropped_samples: int = 0
    turns_with_img: int = 0
    missing_media_files: int = 0
    failed_webm_decode: int = 0
    non_last_neg_found: int = 0
    webm_total: int = 0
    webm_converted: int = 0
    linked_or_copied: int = 0


def try_make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_speaker_prefix(text: str, turn_idx: int) -> str:
    """
    Map arbitrary [speakerN] to only [speaker1]/[speaker2] for tokenizer compatibility.
    """
    if not isinstance(text, str):
        return ""
    m = SPEAKER_RE.match(text)
    if m is None:
        fallback = "[speaker1]" if turn_idx % 2 == 0 else "[speaker2]"
        return f"{fallback}{text}"
    speaker_id = int(m.group(1))
    mapped = "[speaker1]" if speaker_id % 2 == 1 else "[speaker2]"
    return mapped + text[m.end() :]


def infer_img_label(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    return name


def safe_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_media_ref(name: str) -> str:
    """
    Normalize media reference to a filename usable under media_root.
    Some sources contain prefixed folders like "12345/67890.gif".
    """
    return os.path.basename(name.strip())


def make_target_filename(src_name: str) -> str:
    stem, ext = os.path.splitext(src_name)
    if ext.lower() == ".webm":
        return f"{stem}.png"
    return src_name


def copy_or_link(src_path: str, dst_path: str) -> None:
    """Copy src to dst. Skips if dst already exists as a regular file."""
    if os.path.lexists(dst_path):
        if os.path.islink(dst_path):
            os.unlink(dst_path)  # Replace symlinks with real copies
        elif os.path.isfile(dst_path):
            return  # Already have real file; skip
        else:
            os.unlink(dst_path)
    shutil.copy2(src_path, dst_path)


def convert_webm_to_png_first_frame(src_path: str, dst_path: str) -> bool:
    """
    Best-effort decoding:
      1) cv2
      2) imageio
      3) ffmpeg CLI
    """
    # 1) OpenCV
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(src_path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return bool(cv2.imwrite(dst_path, frame))
    except Exception:
        pass

    # 2) imageio
    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(src_path, index=0)
        iio.imwrite(dst_path, frame)
        return True
    except Exception:
        pass

    # 3) ffmpeg command
    try:
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            return False
        src_abs = os.path.abspath(src_path)
        dst_abs = os.path.abspath(dst_path)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel", "error",
            "-i",
            src_abs,
            "-frames:v",
            "1",
            dst_abs,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return proc.returncode == 0 and os.path.isfile(dst_abs)
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert u-sticker json to Sticker-Selection train_pair format.")
    parser.add_argument(
        "--input-json",
        default="./u-sticker/all_filtered_sticker_output.json",
        help="Input u-sticker json path.",
    )
    parser.add_argument(
        "--media-root",
        default="./u-sticker/final_stickers",
        help="Directory containing original media files named by img_id strings.",
    )
    parser.add_argument(
        "--output-train-pair",
        default="./u-sticker/u_sticker_train_pair.json",
        help="Output converted train_pair json path.",
    )
    parser.add_argument(
        "--output-id2img",
        default="./u-sticker/u_sticker_id2img.json",
        help="Output id2img mapping path.",
    )
    parser.add_argument(
        "--output-media-dir",
        default="./u-sticker/u_sticker_media",
        help="Output directory used by --img_dir during training.",
    )
    parser.add_argument(
        "--normalize-speakers",
        action="store_true",
        help="Normalize speaker tags to only [speaker1]/[speaker2].",
    )
    parser.add_argument(
        "--keep-history-img",
        action="store_true",
        help="Keep historical turns' img_id fields (default drops to null except last turn).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for webm-to-PNG conversion (default: 8).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = ConvertStats()

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    if not isinstance(raw_samples, list):
        raise ValueError("Input JSON must be a list of samples.")

    stats.total_samples = len(raw_samples)
    try_make_dir(args.output_media_dir)
    try_make_dir(os.path.dirname(os.path.abspath(args.output_train_pair)))
    try_make_dir(os.path.dirname(os.path.abspath(args.output_id2img)))

    # Gather all media names referenced by img_id / neg_img_id.
    referenced_names: Set[str] = set()
    for sample in raw_samples:
        dialog = sample.get("dialog", [])
        if not isinstance(dialog, list):
            continue
        for turn in dialog:
            if not isinstance(turn, dict):
                continue
            for key in ("img_id", "neg_img_id"):
                name = safe_str(turn.get(key))
                if name is not None:
                    referenced_names.add(normalize_media_ref(name))

    src_to_target: Dict[str, str] = {}
    for src_name in sorted(referenced_names):
        src_to_target[src_name] = make_target_filename(src_name)

    # Materialize media directory: non-webm first (fast), then webm in parallel.
    webm_tasks: List[Tuple[str, str]] = []
    for src_name, target_name in src_to_target.items():
        src_path = os.path.join(args.media_root, src_name)
        dst_path = os.path.join(args.output_media_dir, target_name)
        if not os.path.exists(src_path):
            stats.missing_media_files += 1
            continue
        ext = os.path.splitext(src_name)[1].lower()
        if ext == ".webm":
            stats.webm_total += 1
            if not os.path.exists(dst_path):
                webm_tasks.append((src_path, dst_path))
            continue
        # Copy (replace symlinks with real files)
        if not os.path.isfile(dst_path) or os.path.islink(dst_path):
            copy_or_link(src_path, dst_path)
            stats.linked_or_copied += 1

    # Convert webm to PNG in parallel.
    workers = max(1, getattr(args, "workers", 8))
    if webm_tasks:
        print(f"[convert] Webm->PNG: {len(webm_tasks)} files, {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(convert_webm_to_png_first_frame, s, d): (s, d) for s, d in webm_tasks}
            for fut in as_completed(futures):
                ok = fut.result()
                if ok:
                    stats.webm_converted += 1
                else:
                    stats.failed_webm_decode += 1

    # Build numeric id mapping from actually available media files only.
    all_targets = set(src_to_target.values())
    available_targets = sorted(
        [name for name in all_targets if os.path.exists(os.path.join(args.output_media_dir, name))]
    )
    target_to_int_id: Dict[str, int] = {name: i for i, name in enumerate(available_targets)}
    id2img = {str(i): name for i, name in enumerate(available_targets)}
    unavailable_target_count = len(all_targets) - len(available_targets)

    converted_samples: List[Dict] = []
    for idx, sample in enumerate(raw_samples):
        dialog = sample.get("dialog", [])
        if not isinstance(dialog, list) or len(dialog) == 0:
            stats.dropped_samples += 1
            continue

        new_dialog = []
        for turn_idx, turn in enumerate(dialog):
            if not isinstance(turn, dict):
                continue
            text = turn.get("text", "")
            if args.normalize_speakers:
                text = normalize_speaker_prefix(text, turn_idx=turn_idx)

            img_name = safe_str(turn.get("img_id"))
            neg_name = safe_str(turn.get("neg_img_id"))
            if img_name is not None:
                img_name = normalize_media_ref(img_name)
            if neg_name is not None:
                neg_name = normalize_media_ref(neg_name)

            # Only keep history img optionally; always keep last-turn img.
            is_last = turn_idx == len(dialog) - 1
            keep_turn_img = is_last or args.keep_history_img
            img_id = None
            img_label = None
            if keep_turn_img and img_name is not None:
                target = src_to_target.get(img_name)
                if target is not None and target in target_to_int_id:
                    img_id = target_to_int_id[target]
                    img_label = infer_img_label(target)
                    stats.turns_with_img += 1

            emotion_id = turn.get("emotion_id", None)
            if img_id is not None and emotion_id is None:
                emotion_id = -100

            out_turn = {
                "text": text,
                "img_id": img_id,
                "img_label": img_label,
                "emotion_id": emotion_id,
            }

            if not is_last and neg_name is not None:
                stats.non_last_neg_found += 1

            if is_last:
                if img_name is None or neg_name is None:
                    out_turn = None
                else:
                    pos_target = src_to_target.get(img_name)
                    neg_target = src_to_target.get(neg_name)
                    if (
                        pos_target is None
                        or neg_target is None
                        or pos_target not in target_to_int_id
                        or neg_target not in target_to_int_id
                    ):
                        out_turn = None
                    else:
                        pos_id = target_to_int_id[pos_target]
                        neg_id = target_to_int_id[neg_target]
                        if pos_id == neg_id:
                            out_turn = None
                        else:
                            out_turn["img_id"] = pos_id
                            out_turn["img_label"] = infer_img_label(pos_target)
                            out_turn["neg_img_id"] = neg_id
                            out_turn["neg_img_label"] = infer_img_label(neg_target)
                            if out_turn["emotion_id"] is None:
                                out_turn["emotion_id"] = -100

            if out_turn is None:
                new_dialog = None
                break
            new_dialog.append(out_turn)

        if not new_dialog:
            stats.dropped_samples += 1
            continue

        dialogue_id = sample.get("dialogue_id", f"u_sticker_dialogue_{idx}")
        # If user_id is missing, keep it compatible with project fallback convention.
        last_text = new_dialog[-1]["text"] if new_dialog else "[speaker1]"
        speaker_tag = "[speaker1]"
        m = SPEAKER_RE.match(last_text)
        if m is not None:
            speaker_tag = "[speaker1]" if int(m.group(1)) % 2 == 1 else "[speaker2]"
        user_id = sample.get("user_id", f"{dialogue_id}::{speaker_tag}")

        converted_samples.append(
            {
                "dialog": new_dialog,
                "user_id": user_id,
                "dialogue_id": str(dialogue_id),
            }
        )
        stats.kept_samples += 1

    with open(args.output_train_pair, "w", encoding="utf-8") as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)

    with open(args.output_id2img, "w", encoding="utf-8") as f:
        json.dump(id2img, f, ensure_ascii=False, indent=2)

    print("\n[Convert Summary]")
    print(f"- total_samples: {stats.total_samples}")
    print(f"- kept_samples: {stats.kept_samples}")
    print(f"- dropped_samples: {stats.dropped_samples}")
    print(f"- turns_with_img_kept: {stats.turns_with_img}")
    print(f"- unique_media_targets_available: {len(available_targets)}")
    print(f"- unique_media_targets_unavailable: {unavailable_target_count}")
    print(f"- missing_media_files: {stats.missing_media_files}")
    print(f"- webm_total: {stats.webm_total}")
    print(f"- webm_converted: {stats.webm_converted}")
    print(f"- failed_webm_decode: {stats.failed_webm_decode}")
    print(f"- linked_or_copied_non_webm: {stats.linked_or_copied}")
    print(f"- non_last_neg_found: {stats.non_last_neg_found}")
    print("\n[Outputs]")
    print(f"- train_pair: {args.output_train_pair}")
    print(f"- id2img: {args.output_id2img}")
    print(f"- media_dir: {args.output_media_dir}")
    print("\n[Training Args Hint]")
    print(
        "Use these overrides: "
        f"--train_data_path {args.output_train_pair} "
        f"--id2img_path {args.output_id2img} "
        f"--img_dir {args.output_media_dir} "
        f"--max_image_id {len(available_targets)}"
    )


if __name__ == "__main__":
    main()
