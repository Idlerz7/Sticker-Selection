"""Stage-A audit statistics and preregistered gate helpers."""

from __future__ import annotations

import collections
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .lpc import exact_topk, exact_topk_with_group_filtered
from .metrics import jaccard_rows, paired_bootstrap_delta


def direct_train_index_neighbors(features, ids, train_ids, k=10, device="cuda:0", query_batch_size=2048, index_batch_size=8192):
    row_by_id = {int(value): row for row, value in enumerate(ids.tolist())}
    train_rows = torch.tensor([row_by_id[int(value)] for value in train_ids.tolist()])
    train_features = features[train_rows].to(device)
    train_neighbors, _ = exact_topk(
        train_features, train_features, train_ids.to(device), k, query_ids=train_ids.to(device), exclude_equal_id=True,
        query_batch_size=query_batch_size, index_batch_size=index_batch_size,
    )
    output = torch.full((len(ids), k), -1, dtype=torch.long)
    output[train_rows] = train_neighbors
    mask = torch.ones(len(ids), dtype=torch.bool); mask[train_rows] = False
    if mask.any():
        query_neighbors, _ = exact_topk(
            features[mask].to(device), train_features, train_ids.to(device), k,
            query_batch_size=query_batch_size, index_batch_size=index_batch_size,
        )
        output[mask] = query_neighbors
    return output


def full_neighbors(features, ids, k, device="cuda:0", query_batch_size=2048, index_batch_size=8192):
    return exact_topk(
        features.to(device), features.to(device), ids.to(device), k,
        query_ids=ids.to(device), exclude_equal_id=True,
        query_batch_size=query_batch_size, index_batch_size=index_batch_size,
    )[0]


def full_neighbors_and_duplicate_filtered(features, ids, canonical_hashes, k=10, device="cuda:0", query_batch_size=2048, index_batch_size=8192):
    group_by_hash = {}
    labels = []
    for value in canonical_hashes:
        if value not in group_by_hash: group_by_hash[value] = len(group_by_hash)
        labels.append(group_by_hash[value])
    labels = torch.tensor(labels, dtype=torch.long)
    values = features.to(device)
    return exact_topk_with_group_filtered(
        values, values, ids.to(device), labels.to(device), labels.to(device), k,
        query_ids=ids.to(device), exclude_equal_id=True,
        query_batch_size=query_batch_size, index_batch_size=index_batch_size,
    )


def filter_duplicate_neighbors(neighbors, ids, canonical_hashes, k=10):
    output = torch.full((len(ids), k), -1, dtype=torch.long)
    hash_by_id = {int(image_id): canonical_hashes[row] for row, image_id in enumerate(ids.tolist())}
    for row, (image_id, candidates) in enumerate(zip(ids.tolist(), neighbors.tolist())):
        kept = [int(value) for value in candidates if hash_by_id[int(value)] != hash_by_id[int(image_id)]][:k]
        output[row, :len(kept)] = torch.tensor(kept, dtype=torch.long)
    return output


def pack_recall(neighbors, ids, pack_by_id, ks=(1, 5, 10)):
    by_pack = collections.defaultdict(list)
    per_k = {k: [] for k in ks}
    for image_id, row in zip(ids.tolist(), neighbors.tolist()):
        pack = str(pack_by_id[int(image_id)])
        candidates = [int(value) for value in row if int(value) >= 0]
        for k in ks:
            hit = float(any(str(pack_by_id[value]) == pack for value in candidates[:k]))
            per_k[k].append(hit); by_pack[(pack, k)].append(hit)
    ordinary = {"r@%d" % k: float(np.mean(per_k[k])) for k in ks}
    macro = {"r@%d" % k: float(np.mean([np.mean(values) for (pack, kk), values in by_pack.items() if kk == k])) for k in ks}
    pack_vectors = {k: {pack: float(np.mean(values)) for (pack, kk), values in by_pack.items() if kk == k} for k in ks}
    return {"ordinary": ordinary, "pack_macro": macro, "per_anchor": per_k, "per_pack": pack_vectors}


def paired_pack_bootstrap(left, right, k=10, replicates=10000, seed=2021):
    packs = sorted(set(left["per_pack"][k]) & set(right["per_pack"][k]))
    return paired_bootstrap_delta(
        [left["per_pack"][k][pack] for pack in packs], [right["per_pack"][k][pack] for pack in packs],
        replicates, seed,
    )


def edge_leakage(neighbors, ids, canonical_hashes, ocr_by_id, pack_by_id=None):
    hash_by_id = {int(image_id): canonical_hashes[row] for row, image_id in enumerate(ids.tolist())}
    total = duplicate = exact_ocr = char_overlap = same_pack = 0
    indegree = collections.Counter()
    for image_id, row in zip(ids.tolist(), neighbors.tolist()):
        source_ocr = "".join(str(ocr_by_id.get(int(image_id), "")).lower().split())
        source_chars = set(source_ocr)
        for target in row:
            target = int(target)
            if target < 0:
                continue
            total += 1; indegree[target] += 1
            duplicate += hash_by_id[int(image_id)] == hash_by_id[target]
            target_ocr = "".join(str(ocr_by_id.get(target, "")).lower().split())
            exact_ocr += bool(source_ocr and target_ocr and source_ocr == target_ocr)
            char_overlap += bool(source_chars and set(target_ocr) and source_chars.intersection(set(target_ocr)))
            if pack_by_id is not None:
                same_pack += str(pack_by_id[int(image_id)]) == str(pack_by_id[target])
    degrees = np.asarray(list(indegree.values()) or [0], dtype=np.float64)
    return {
        "edges": total,
        "exact_duplicate_share": duplicate / total if total else 0.0,
        "exact_ocr_share": exact_ocr / total if total else 0.0,
        "character_overlap_share": char_overlap / total if total else 0.0,
        "same_pack_share": same_pack / total if total and pack_by_id is not None else None,
        "cross_pack_share": 1 - same_pack / total if total and pack_by_id is not None else None,
        "hubness": {"max_indegree": int(degrees.max()), "p95_indegree": float(np.quantile(degrees, 0.95)), "covered_targets": len(indegree)},
    }


def _candidate_geometry(candidate_ids, feature_by_id):
    values = torch.stack([feature_by_id[int(value)] for value in candidate_ids]).float()
    cosine = values.matmul(values.T).clamp(-1, 1)
    distances = 1 - cosine
    upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    return values, float(upper.mean()), float(upper.std(unbiased=False))


def a4_probe(pair_cache, candidate_manifest, descriptor_bundles, lpc_bundle, source_rows, replicates=10000, seed=2021):
    scores = pair_cache["b"].float()
    positive = pair_cache["positive_index"].long()
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = (order == positive.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
    target = (ranks == 1).numpy().astype(np.int64)
    probabilities = torch.softmax(scores, dim=1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
    sorted_scores = torch.sort(scores, dim=1, descending=True).values
    controls = torch.stack((
        sorted_scores[:, 0], sorted_scores[:, 0] - sorted_scores[:, 1], entropy,
        scores.mean(dim=1), scores.std(dim=1, unbiased=False), pair_cache["token_lengths"].float(),
    ), dim=1).numpy()
    maps = {
        name: {int(image_id): bundle["features"][row] for row, image_id in enumerate(bundle["ids"].tolist())}
        for name, bundle in descriptor_bundles.items()
    }
    lpc_map = {int(image_id): lpc_bundle["features"][row] for row, image_id in enumerate(lpc_bundle["ids"].tolist())}
    degree_map = {int(image_id): int(lpc_bundle["degree"][row]) for row, image_id in enumerate(lpc_bundle["ids"].tolist())}
    fallback_map = {int(image_id): bool(lpc_bundle["fallback"][row]) for row, image_id in enumerate(lpc_bundle["ids"].tolist())}
    presentation = []
    records = []
    for query_index, manifest in enumerate(candidate_manifest):
        candidates = [int(value) for value in manifest["candidate_ids"]]
        predicted = int(order[query_index, 0])
        gold = int(positive[query_index])
        row_features = []
        descriptor_record = {}
        for name, mapping in {**maps, "lpc": lpc_map}.items():
            values, mean_distance, std_distance = _candidate_geometry(candidates, mapping)
            centroid = torch.nn.functional.normalize(values.mean(dim=0), dim=0)
            pred_centroid = float(1 - torch.dot(values[predicted], centroid))
            gold_pred = float(1 - torch.dot(values[gold], values[predicted]))
            row_features.extend((mean_distance, std_distance, pred_centroid))
            descriptor_record[name] = {
                "candidate_pair_distance_mean": mean_distance, "candidate_pair_distance_std": std_distance,
                "predicted_to_centroid_distance": pred_centroid, "gold_to_predicted_distance_descriptive_only": gold_pred,
            }
        predicted_id = candidates[predicted]
        candidate_degrees = [degree_map[value] for value in candidates]
        row_features.extend((float(np.mean(candidate_degrees)), float(np.std(candidate_degrees)), degree_map[predicted_id], float(fallback_map[predicted_id])))
        presentation.append(row_features)
        records.append({
            "query_id": manifest["query_id"], "source_row": manifest["source_row"], "dialogue_id": manifest["dialogue_id"],
            "candidate_ids": candidates, "positive_index": gold, "base_logits": scores[query_index].tolist(),
            "rank": int(ranks[query_index]), "margin": float(sorted_scores[query_index, 0] - sorted_scores[query_index, 1]),
            "correct_at_1": bool(target[query_index]), "descriptors": descriptor_record,
            "lpc_density": {"candidate_degree_mean": float(np.mean(candidate_degrees)), "candidate_degree_std": float(np.std(candidate_degrees)), "predicted_degree": degree_map[predicted_id], "predicted_fallback": fallback_map[predicted_id]},
        })
    presentation = np.asarray(presentation, dtype=np.float64)
    groups = np.asarray([row["dialogue_id"] for row in candidate_manifest])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(splitter.split(controls, target, groups))
    control_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=seed, max_iter=2000))
    presentation_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=seed, max_iter=2000))
    control_model.fit(controls[train], target[train])
    presentation_model.fit(np.concatenate((controls, presentation), axis=1)[train], target[train])
    control_pred = control_model.predict_proba(controls[test])[:, 1]
    presentation_pred = presentation_model.predict_proba(np.concatenate((controls, presentation), axis=1)[test])[:, 1]
    control_auc = roc_auc_score(target[test], control_pred)
    presentation_auc = roc_auc_score(target[test], presentation_pred)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(replicates):
        sampled = rng.randint(0, len(test), size=len(test))
        labels = target[test][sampled]
        if len(np.unique(labels)) < 2:
            continue
        deltas.append(roc_auc_score(labels, presentation_pred[sampled]) - roc_auc_score(labels, control_pred[sampled]))
    ci = [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
    gain = float(presentation_auc - control_auc)
    return records, {
        "split": {"method": "GroupShuffleSplit", "train_queries": len(train), "heldout_queries": len(test), "seed": seed, "test_size": 0.2},
        "control_features": ["max_logit", "top1_top2_margin", "entropy", "logit_mean", "logit_std", "token_length"],
        "presentation_feature_count": presentation.shape[1],
        "control_auroc": float(control_auc), "control_plus_presentation_auroc": float(presentation_auc),
        "auroc_gain": gain, "bootstrap_ci95": ci, "bootstrap_replicates": len(deltas),
        "incremental_support": bool(gain >= 0.01 and ci[0] > 0),
        "gold_to_predicted_distance_used_in_classifier": False,
    }
