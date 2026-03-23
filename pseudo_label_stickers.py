#!/usr/bin/env python3
import argparse
import base64
import datetime
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib import error, request
from PIL import Image


DEFAULT_PROMPT = """You are helping build weak supervision for a conversational sticker retrieval dataset.

Your task is to identify the sticker's stable visual identity and template/style.

Important:
- Do NOT guess any original sticker pack name.
- Do NOT focus on the text meaning in the sticker.
- Ignore text semantics as much as possible.
- Focus on stable visual template/identity/style only.
- "main_subject" must describe only the core entity/template itself, not its current action, posture, or emotion.

Return JSON only in the following format:

{
  "main_subject": "",
  "subject_category": "",
  "visual_style": "",
  "identity_summary": ""
}

Field instructions:

1. "main_subject"
- Use a short normalized phrase for the core entity/template only.
- Do NOT include action, posture, or emotion words.
- Examples:
  "panda_face", "mushroom_head_character", "rabbit_character", "white_blob_face", "cat_meme", "real_child_face", "anime_girl"

2. "subject_category"
- Use a coarse reusable category with a short normalized phrase.
- Prefer broad visual identity categories rather than very specific labels.
- Examples include:
  "animal_character", "human_character", "meme_face", "anime_character", "simple_doodle_character", "real_human", "real_animal", "emoji_icon", "object_character"

3. "visual_style"
- Use a short normalized phrase for the overall visual template/style family.
- Prefer broad reusable style families rather than dataset-specific names.
- Examples include:
  "black_white_reaction_meme", "simple_line_doodle", "cute_cartoon_sticker", "real_photo_meme", "anime_portrait", "emoji_icon_style"

4. "identity_summary"
- Write one short sentence summarizing the stable template identity and visual style only.
- Do NOT describe transient expression, action, or text meaning.
- Example:
  "a panda face in black-white reaction meme style"
  "a rabbit character in cute cartoon sticker style"

Rules:
1. Output valid JSON only.
2. Keep all values short, normalized, and reusable.
3. Do not invent pack names.
4. Do not provide extra explanation.
5. Prefer stable visual identity/style over transient expression."""

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo labels for stickers in data/meme_set via OpenAI-compatible API."
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://xh.v1api.cc"))
    parser.add_argument(
        "--api-path", default=os.getenv("OPENAI_API_PATH", "/v1/chat/completions")
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--image-dir",
        default="data/meme_set",
        help="Directory containing sticker images.",
    )
    parser.add_argument(
        "--img2id-path",
        default="data/img2id.json",
        help="Path to img2id mapping. If missing, filename stem is used as id.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--errors-output",
        default="",
        help="Failed samples output JSONL path.",
    )
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Optional prompt txt file path; if not set, use built-in prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process first N images (0 means all)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess all records and overwrite existing output files.",
    )
    parser.add_argument(
        "--convert-gif-first-frame",
        action="store_true",
        default=True,
        help="Convert GIF input to first-frame PNG before upload (recommended for Gemini routes).",
    )
    return parser.parse_args()


def mime_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jpg":
        return "image/jpeg"
    if suffix in {".jpeg", ".png", ".gif", ".webp"}:
        return f"image/{suffix[1:]}"
    raise ValueError(f"Unsupported suffix: {suffix}")


def encode_image_for_api(image_path: Path, convert_gif_first_frame: bool) -> Tuple[str, str]:
    suffix = image_path.suffix.lower()
    if suffix == ".gif" and convert_gif_first_frame:
        with Image.open(image_path) as img:
            img.seek(0)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return payload, "image/png"

    img_bytes = image_path.read_bytes()
    mime = mime_from_suffix(image_path)
    payload = base64.b64encode(img_bytes).decode("ascii")
    return payload, mime


def load_prompt(prompt_file: str) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    return DEFAULT_PROMPT


def load_img2id(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def load_processed_ids(output_path: Path) -> Set[str]:
    processed = set()
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = str(obj.get("id", ""))
            if item_id:
                processed.add(item_id)
    return processed


def list_images(image_dir: Path) -> List[Path]:
    paths = []
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == ".DS_Store":
            continue
        if p.suffix.lower() in ALLOWED_SUFFIXES:
            paths.append(p)
    return sorted(paths, key=lambda x: x.name)


def extract_first_json(text: str) -> Dict:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model output.")
    candidate = match.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def validate_label(label: Dict) -> Dict:
    required = {"main_subject", "subject_category", "visual_style", "identity_summary"}
    missing = [k for k in required if k not in label]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    normalized = {}
    for k in required:
        val = label.get(k, "")
        if not isinstance(val, str):
            val = str(val)
        normalized[k] = val.strip()

    category = re.sub(r"\s+", "_", normalized["subject_category"].lower())
    category = re.sub(r"[^a-z0-9_]+", "", category)
    if not category:
        category = "other"
    if category == "simple_blob_character":
        category = "simple_doodle_character"
    normalized["subject_category"] = category
    return normalized


def build_payload(model: str, prompt: str, data_url: str, temperature: float, max_tokens: int) -> Dict:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a careful visual labeling assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }


def chat_completion(
    endpoint: str,
    api_key: str,
    payload: Dict,
    timeout: int,
) -> Tuple[str, Dict]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    response_json = json.loads(raw)
    choice = response_json.get("choices", [{}])[0]
    message = choice.get("message", {}).get("content", "")
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        message = "\n".join(parts).strip()
    if not isinstance(message, str) or not message.strip():
        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            raise ValueError(
                "Invalid model content: empty assistant content with finish_reason=length. "
                "Try increasing --max-tokens."
            )
        raise ValueError(f"Invalid model content: {response_json}")
    return message, response_json


def classify_http_error(exc: error.HTTPError) -> Tuple[bool, str]:
    details = ""
    try:
        details = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        details = ""

    merged = f"{exc} {details}".lower()
    non_retry_signals = [
        "mime type is not supported",
        "convert_request_failed",
        "invalid_request",
        "unsupported",
    ]
    retryable = not any(sig in merged for sig in non_retry_signals)
    return retryable, details


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_model_for_filename(model_name: str) -> str:
    # Keep filename readable and safe across filesystems.
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip())
    safe = safe.strip("._-")
    return safe or "unknown_model"


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError(
            "API key is empty. Please set --api-key or OPENAI_API_KEY."
        )
    if "你的key" in args.api_key or "your_key" in args.api_key.lower():
        raise ValueError(
            "API key looks like a placeholder value. Please provide a real key."
        )
    try:
        args.api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "API key contains non-ASCII characters. "
            "This often happens when placeholder text is used by mistake."
        ) from exc

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    model_for_name = sanitize_model_for_filename(args.model)
    date_for_name = datetime.datetime.now().strftime("%Y%m%d")
    default_output = (
        f"pseudo_labels/sticker_identity_style_labels_from_data_meme_set_"
        f"model_{model_for_name}_date_{date_for_name}.jsonl"
    )
    default_errors = (
        f"pseudo_labels/sticker_identity_style_labels_from_data_meme_set_"
        f"model_{model_for_name}_date_{date_for_name}_errors.jsonl"
    )

    output_path = Path(args.output) if args.output else Path(default_output)
    errors_path = Path(args.errors_output) if args.errors_output else Path(default_errors)
    ensure_parent(output_path)
    ensure_parent(errors_path)

    prompt = load_prompt(args.prompt_file)
    img2id = load_img2id(args.img2id_path)
    images = list_images(image_dir)
    if args.limit > 0:
        images = images[: args.limit]

    if args.overwrite:
        processed_ids = set()
        output_mode = "w"
        errors_mode = "w"
    else:
        processed_ids = load_processed_ids(output_path)
        output_mode = "a"
        errors_mode = "a"

    endpoint = f"{args.base_url.rstrip('/')}/{args.api_path.lstrip('/')}"

    total = len(images)
    success = 0
    skipped = 0
    failed = 0

    with output_path.open(output_mode, encoding="utf-8") as out_f, errors_path.open(
        errors_mode, encoding="utf-8"
    ) as err_f:
        for idx, image_path in enumerate(images, start=1):
            filename = image_path.name
            image_id = img2id.get(filename, image_path.stem)

            if not args.overwrite and image_id in processed_ids:
                skipped += 1
                print(f"[{idx}/{total}] skip id={image_id} file={filename}")
                continue

            print(f"[{idx}/{total}] processing id={image_id} file={filename}")
            b64, mime = encode_image_for_api(
                image_path=image_path,
                convert_gif_first_frame=args.convert_gif_first_frame,
            )
            data_url = f"data:{mime};base64,{b64}"
            payload = build_payload(
                model=args.model,
                prompt=prompt,
                data_url=data_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            ok = False
            last_error = ""
            for attempt in range(1, args.retries + 1):
                try:
                    content, _ = chat_completion(
                        endpoint=endpoint,
                        api_key=args.api_key,
                        payload=payload,
                        timeout=args.timeout,
                    )
                    label = validate_label(extract_first_json(content))
                    record = {
                        "id": image_id,
                        "filename": filename,
                        "label": label,
                        "model": args.model,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success += 1
                    ok = True
                    break
                except (ValueError, json.JSONDecodeError, error.HTTPError, error.URLError) as exc:
                    last_error = str(exc)
                    retryable = True
                    http_details = ""
                    if isinstance(exc, error.HTTPError):
                        retryable, http_details = classify_http_error(exc)
                        if http_details:
                            last_error = f"{last_error} | {http_details}"
                    print(
                        f"  attempt {attempt}/{args.retries} failed: {last_error}",
                        flush=True,
                    )
                    if http_details:
                        print(f"  http_error_body: {http_details}", flush=True)
                    if (not retryable) or attempt >= args.retries:
                        break
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep * attempt)

            if not ok:
                failed += 1
                err_record = {
                    "id": image_id,
                    "filename": filename,
                    "error": last_error,
                }
                err_f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                err_f.flush()

    print("=" * 72)
    print(f"endpoint: {endpoint}")
    print(f"model: {args.model}")
    print(f"total: {total}, success: {success}, skipped: {skipped}, failed: {failed}")
    print(f"output: {output_path}")
    print(f"errors: {errors_path}")


if __name__ == "__main__":
    main()
