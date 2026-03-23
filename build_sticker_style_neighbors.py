#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class NeighborConfig:
    input_path: str = (
        "pseudo_labels/"
        "sticker_identity_style_labels_from_data_meme_set_model_gemini-3-pro-preview_date_20260321.jsonl"
    )
    output_path: str = "style_neighbors.json"
    embedding_model: str = (
        "models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    top_k: int = 5
    batch_size: int = 64

    # Candidate filtering: at least one condition if enabled.
    require_same_subject_category_or_style: bool = True
    fallback_to_all_when_too_few_candidates: bool = True
    min_candidates_before_fallback: int = 3

    # Weighted multi-signal scoring.
    w_full: float = 0.50
    w_subject: float = 0.20
    w_style: float = 0.20
    w_same_category: float = 0.05
    w_same_style: float = 0.05

    normalize_embeddings: bool = True
    save_config: bool = True
    allow_local_hash_fallback: bool = True
    hash_embedding_dim: int = 1024


def parse_args() -> NeighborConfig:
    parser = argparse.ArgumentParser(
        description="Build top-k sticker style neighbors from pseudo-label JSONL."
    )
    parser.add_argument("--input-path", default=NeighborConfig.input_path)
    parser.add_argument("--output-path", default=NeighborConfig.output_path)
    parser.add_argument("--embedding-model", default=NeighborConfig.embedding_model)
    parser.add_argument("--top-k", type=int, default=NeighborConfig.top_k)
    parser.add_argument("--batch-size", type=int, default=NeighborConfig.batch_size)
    parser.add_argument(
        "--require-same-subject-category-or-style",
        action="store_true",
        default=NeighborConfig.require_same_subject_category_or_style,
    )
    parser.add_argument(
        "--no-require-same-subject-category-or-style",
        dest="require_same_subject_category_or_style",
        action="store_false",
    )
    parser.add_argument(
        "--fallback-to-all-when-too-few-candidates",
        action="store_true",
        default=NeighborConfig.fallback_to_all_when_too_few_candidates,
    )
    parser.add_argument(
        "--no-fallback-to-all-when-too-few-candidates",
        dest="fallback_to_all_when_too_few_candidates",
        action="store_false",
    )
    parser.add_argument(
        "--min-candidates-before-fallback",
        type=int,
        default=NeighborConfig.min_candidates_before_fallback,
    )
    parser.add_argument("--w-full", type=float, default=NeighborConfig.w_full)
    parser.add_argument("--w-subject", type=float, default=NeighborConfig.w_subject)
    parser.add_argument("--w-style", type=float, default=NeighborConfig.w_style)
    parser.add_argument(
        "--w-same-category", type=float, default=NeighborConfig.w_same_category
    )
    parser.add_argument("--w-same-style", type=float, default=NeighborConfig.w_same_style)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=NeighborConfig.normalize_embeddings,
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
    )
    parser.add_argument("--save-config", action="store_true", default=NeighborConfig.save_config)
    parser.add_argument("--no-save-config", dest="save_config", action="store_false")
    parser.add_argument(
        "--allow-local-hash-fallback",
        action="store_true",
        default=NeighborConfig.allow_local_hash_fallback,
    )
    parser.add_argument(
        "--no-allow-local-hash-fallback",
        dest="allow_local_hash_fallback",
        action="store_false",
    )
    parser.add_argument("--hash-embedding-dim", type=int, default=NeighborConfig.hash_embedding_dim)

    args = parser.parse_args()
    cfg = NeighborConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        require_same_subject_category_or_style=args.require_same_subject_category_or_style,
        fallback_to_all_when_too_few_candidates=args.fallback_to_all_when_too_few_candidates,
        min_candidates_before_fallback=args.min_candidates_before_fallback,
        w_full=args.w_full,
        w_subject=args.w_subject,
        w_style=args.w_style,
        w_same_category=args.w_same_category,
        w_same_style=args.w_same_style,
        normalize_embeddings=args.normalize_embeddings,
        save_config=args.save_config,
        allow_local_hash_fallback=args.allow_local_hash_fallback,
        hash_embedding_dim=args.hash_embedding_dim,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: NeighborConfig) -> None:
    if cfg.top_k <= 0:
        raise ValueError("--top-k must be > 0.")
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if cfg.min_candidates_before_fallback < 0:
        raise ValueError("--min-candidates-before-fallback must be >= 0.")
    if cfg.hash_embedding_dim <= 0:
        raise ValueError("--hash-embedding-dim must be > 0.")
    for name in ("w_full", "w_subject", "w_style", "w_same_category", "w_same_style"):
        value = getattr(cfg, name)
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    weight_sum = cfg.w_full + cfg.w_subject + cfg.w_style + cfg.w_same_category + cfg.w_same_style
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        print(f"[WARN] Weights sum to {weight_sum:.4f} (not 1.0). This is allowed.")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse failed at line {line_no}: {exc}") from exc
            items.append(obj)
    if not items:
        raise ValueError("Input JSONL is empty after parsing.")
    return items


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    text = str(s).strip().lower()
    text = text.replace("-", "_")
    text = " ".join(text.split())
    return text


def normalize_label_record(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    label = item.get("label", {})
    if not isinstance(label, dict):
        label = {}
    out = {
        "id": str(item.get("id", "")).strip(),
        "filename": str(item.get("filename", "")).strip(),
        "model": str(item.get("model", "")).strip(),
        "main_subject": normalize_text(label.get("main_subject", "")),
        "subject_category": normalize_text(label.get("subject_category", "")),
        "visual_style": normalize_text(label.get("visual_style", "")),
        "identity_summary": normalize_text(label.get("identity_summary", "")),
        "_row_idx": idx,
    }
    if not out["id"]:
        out["id"] = f"row_{idx}"
    return out


def deduplicate_by_id(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for item in data:
        item_id = item["id"]
        if item_id in seen:
            duplicates += 1
        seen[item_id] = item
    if duplicates > 0:
        print(f"[WARN] Found {duplicates} duplicate ids. Keeping the last occurrence.")
    return list(seen.values())


def build_text_views(x: Dict[str, Any]) -> Dict[str, str]:
    subject_text = f"{x['main_subject']} [SEP] {x['subject_category']}".strip()
    style_text = f"{x['visual_style']} [SEP] {x['subject_category']}".strip()
    full_text = (
        f"{x['main_subject']} [SEP] {x['subject_category']} [SEP] "
        f"{x['visual_style']} [SEP] {x['identity_summary']}"
    ).strip()
    return {
        "subject_text": subject_text,
        "style_text": style_text,
        "full_text": full_text,
    }


def encode_texts(model: SentenceTransformer, texts: List[str], cfg: NeighborConfig) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embedding shape is invalid: {emb.shape}")
    return emb


def _tokenize_for_hash(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return text.split(" ")


def encode_texts_local_hash(texts: List[str], cfg: NeighborConfig) -> np.ndarray:
    dim = cfg.hash_embedding_dim
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, total=len(texts), desc="Hash-encoding")):
        tokens = _tokenize_for_hash(text)
        if not tokens:
            continue
        for tok in tokens:
            idx = hash(tok) % dim
            out[i, idx] += 1.0
    # L2 normalize so cosine is valid with dot-product.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    out = out / norms
    return out


def similarity_matrix(emb: np.ndarray, normalize_embeddings: bool) -> np.ndarray:
    if normalize_embeddings:
        sim = emb @ emb.T
    else:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-8, a_max=None)
        normed = emb / norms
        sim = normed @ normed.T
    return sim.astype(np.float32)


def is_candidate_pair(xi: Dict[str, Any], xj: Dict[str, Any], cfg: NeighborConfig) -> bool:
    if xi["id"] == xj["id"]:
        return False
    if not cfg.require_same_subject_category_or_style:
        return True
    if xi["subject_category"] and xi["subject_category"] == xj["subject_category"]:
        return True
    if xi["visual_style"] and xi["visual_style"] == xj["visual_style"]:
        return True
    return False


def score_pair(
    i: int,
    j: int,
    data: List[Dict[str, Any]],
    sim_subject: np.ndarray,
    sim_style: np.ndarray,
    sim_full: np.ndarray,
    cfg: NeighborConfig,
) -> float:
    xi = data[i]
    xj = data[j]
    same_category = 1.0 if xi["subject_category"] == xj["subject_category"] and xi["subject_category"] else 0.0
    same_style = 1.0 if xi["visual_style"] == xj["visual_style"] and xi["visual_style"] else 0.0
    return (
        cfg.w_full * float(sim_full[i, j])
        + cfg.w_subject * float(sim_subject[i, j])
        + cfg.w_style * float(sim_style[i, j])
        + cfg.w_same_category * same_category
        + cfg.w_same_style * same_style
    )


def build_neighbors(data: List[Dict[str, Any]], cfg: NeighborConfig) -> List[Dict[str, Any]]:
    print(f"[INFO] Loading embedding model: {cfg.embedding_model}")
    model = None
    using_local_hash = False
    try:
        model = SentenceTransformer(cfg.embedding_model)
    except Exception as exc:
        if not cfg.allow_local_hash_fallback:
            raise
        using_local_hash = True
        print(
            "[WARN] Failed to load sentence-transformer model. "
            f"Falling back to local hash embedding. Reason: {exc}"
        )

    views = [build_text_views(x) for x in data]
    subject_texts = [v["subject_text"] for v in views]
    style_texts = [v["style_text"] for v in views]
    full_texts = [v["full_text"] for v in views]

    if using_local_hash:
        print("[INFO] Encoding subject_text with local hash embedding...")
        emb_subject = encode_texts_local_hash(subject_texts, cfg)
        print("[INFO] Encoding style_text with local hash embedding...")
        emb_style = encode_texts_local_hash(style_texts, cfg)
        print("[INFO] Encoding full_text with local hash embedding...")
        emb_full = encode_texts_local_hash(full_texts, cfg)
    else:
        print("[INFO] Encoding subject_text...")
        emb_subject = encode_texts(model, subject_texts, cfg)
        print("[INFO] Encoding style_text...")
        emb_style = encode_texts(model, style_texts, cfg)
        print("[INFO] Encoding full_text...")
        emb_full = encode_texts(model, full_texts, cfg)

    print("[INFO] Computing similarity matrices...")
    sim_subject = similarity_matrix(emb_subject, cfg.normalize_embeddings)
    sim_style = similarity_matrix(emb_style, cfg.normalize_embeddings)
    sim_full = similarity_matrix(emb_full, cfg.normalize_embeddings)

    n = len(data)
    results: List[Dict[str, Any]] = []
    fallback_count = 0

    print("[INFO] Building top-k neighbors...")
    for i in tqdm(range(n), total=n):
        xi = data[i]

        candidate_indices = [j for j in range(n) if is_candidate_pair(xi, data[j], cfg)]
        if (
            cfg.fallback_to_all_when_too_few_candidates
            and len(candidate_indices) < cfg.min_candidates_before_fallback
        ):
            candidate_indices = [j for j in range(n) if j != i]
            fallback_count += 1

        scored: List[Dict[str, Any]] = []
        for j in candidate_indices:
            xj = data[j]
            score = score_pair(i, j, data, sim_subject, sim_style, sim_full, cfg)
            scored.append(
                {
                    "id": xj["id"],
                    "filename": xj["filename"],
                    "score": round(float(score), 6),
                    "main_subject": xj["main_subject"],
                    "subject_category": xj["subject_category"],
                    "visual_style": xj["visual_style"],
                    "identity_summary": xj["identity_summary"],
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_neighbors = scored[: cfg.top_k]

        results.append(
            {
                "id": xi["id"],
                "filename": xi["filename"],
                "main_subject": xi["main_subject"],
                "subject_category": xi["subject_category"],
                "visual_style": xi["visual_style"],
                "identity_summary": xi["identity_summary"],
                "neighbors": top_neighbors,
                "meta": {
                    "candidate_count": len(candidate_indices),
                    "used_fallback_to_all": len(candidate_indices) == (n - 1)
                    and cfg.require_same_subject_category_or_style,
                },
            }
        )

    if fallback_count > 0:
        print(f"[WARN] Fallback-to-all triggered for {fallback_count} samples.")
    if using_local_hash:
        print("[WARN] Output built with local hash embeddings due to model load failure.")
    return results


def save_json(path: str, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = parse_args()

    print(f"[INFO] Reading labels from: {cfg.input_path}")
    raw_items = load_jsonl(cfg.input_path)
    data = [normalize_label_record(item, idx=i) for i, item in enumerate(raw_items)]
    data = deduplicate_by_id(data)
    print(f"[INFO] Loaded samples: {len(data)}")

    neighbors = build_neighbors(data, cfg)

    output_obj: Dict[str, Any] = {"neighbors": neighbors}
    if cfg.save_config:
        output_obj["config"] = asdict(cfg)
        output_obj["stats"] = {
            "num_samples": len(data),
            "top_k": cfg.top_k,
        }

    save_json(cfg.output_path, output_obj)
    print(f"[INFO] Done. Saved to: {cfg.output_path}")


if __name__ == "__main__":
    main()
