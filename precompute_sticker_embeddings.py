#!/usr/bin/env python3
"""
预计算全部贴纸的 CLIP 视觉 embedding 并保存到本地。

用法：
    python precompute_sticker_embeddings.py \\
        --id2img_path ./u-sticker/u_sticker_id2img.json \\
        --img_dir ./u-sticker/u_sticker_media \\
        --max_image_id 34894 \\
        --img_pretrain_path ./ckpt/clip-ViT-B-32 \\
        --output_path ./u-sticker/sticker_clip_embs.pt \\
        --batch_size 512 \\
        --workers 8

训练时加 --img_emb_cache_path ./u-sticker/sticker_clip_embs.pt 即可直接加载。
"""
import os
import json
import argparse
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Work around protobuf C-extension segfault on some environments.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def load_image(id, id2imgpath, img_dir, img_process):
    """加载并预处理单张图像。"""
    img_file = id2imgpath.get(id)
    if img_file is None:
        return None
    path = os.path.join(img_dir, img_file)
    if not os.path.exists(path):
        return None
    from img_utils import pick_single_frame
    with open(path, 'rb') as fin:
        data = fin.read()
    img_obj = img_process(pick_single_frame(data))
    return id, img_obj


def _materialize_sticker_batch(ids_batch, id2imgpath, img_dir, img_process, workers):
    """Decode + CPU preprocess into a single tensor [B,3,224,224] and a valid mask."""
    if not ids_batch:
        return torch.zeros(0, 3, 224, 224), torch.zeros(0, dtype=torch.bool)
    start = ids_batch[0]
    b = len(ids_batch)
    img_slots = [None] * b

    def load_one(i):
        return load_image(i, id2imgpath, img_dir, img_process)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_one, i): i for i in ids_batch}
        for f in as_completed(futures):
            try:
                res = f.result()
                if res is not None:
                    idx, obj = res
                    img_slots[idx - start] = obj
            except Exception:
                pass

    tensors = torch.zeros(b, 3, 224, 224)
    valid = torch.zeros(b, dtype=torch.bool)
    for i, o in enumerate(img_slots):
        if o is not None:
            tensors[i] = o
            valid[i] = True
    return tensors, valid


def _hf_clip_projection_dim(img_clip):
    c = img_clip.model.config
    return int(getattr(c, 'projection_dim', None) or 512)


def main():
    parser = argparse.ArgumentParser(description='Precompute CLIP embeddings for all stickers')
    parser.add_argument('--id2img_path', required=True, help='id to image filename mapping')
    parser.add_argument('--img_dir', required=True, help='image directory')
    parser.add_argument('--max_image_id', type=int, required=True, help='max sticker id (exclusive)')
    parser.add_argument('--img_pretrain_path', default='./ckpt/clip-ViT-B-32', help='CLIP model path')
    parser.add_argument('--output_path', default='./u-sticker/sticker_clip_embs.pt', help='output .pt file')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--local_files_only', action='store_true', default=True)
    parser.add_argument(
        '--device',
        default=None,
        help='Force device: cuda, cuda:0, cpu (default: auto). '
        'If you set CUDA_VISIBLE_DEVICES=1, the process still uses cuda:0 — watch physical GPU 1 in nvitop.',
    )
    args = parser.parse_args()

    if args.local_files_only:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'

    # Load id2imgpath
    with open(args.id2img_path, encoding='utf-8') as f:
        id2img_raw = json.load(f)
    id2imgpath = {int(k): v for k, v in id2img_raw.items()}

    # Load CLIP
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import batch_to_device
    from main import HFClipSentenceEncoder

    try:
        img_clip = SentenceTransformer(args.img_pretrain_path)
    except Exception as e:
        print(f"SentenceTransformer failed: {e}, trying HF CLIP...")
        img_clip = HFClipSentenceEncoder(
            args.img_pretrain_path,
            local_files_only=args.local_files_only,
            device=args.device,
        )

    if hasattr(img_clip, 'preprocess'):
        img_process = img_clip.preprocess
    else:
        img_process = img_clip._first_module().preprocess

    if args.device is not None:
        device = torch.device(args.device)
        img_clip = img_clip.to(device)
    else:
        device = getattr(img_clip, '_target_device', None)
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            img_clip = img_clip.to(device)

    print(
        f'[precompute] torch={torch.__version__} cuda_available={torch.cuda.is_available()} '
        f'encode_device={device}'
    )
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        print(
            f'[precompute] cuda:{idx} -> {torch.cuda.get_device_name(idx)} '
            f'(CUDA_VISIBLE_DEVICES remaps visible GPUs; in nvitop match the PID to the physical GPU)'
        )
        torch.backends.cudnn.benchmark = True

    img_clip.eval()

    all_embs = []
    n_total = args.max_image_id
    batch_size = args.batch_size
    workers = args.workers
    use_hf_fast = isinstance(img_clip, HFClipSentenceEncoder)

    if use_hf_fast:
        print(
            '[precompute] HF CLIP: overlapped next-batch CPU decode while GPU encodes current; '
            'wall time is still dominated by PNG decode — nvitop SM% may stay near 0 because '
            'ViT-B/32 forward is short vs I/O.'
        )

    def load_one(i):
        return load_image(i, id2imgpath, args.img_dir, img_process)

    with torch.no_grad():
        if use_hf_fast:
            dim = _hf_clip_projection_dim(img_clip)
            batch_ranges = [
                (s, min(s + batch_size, n_total))
                for s in range(0, n_total, batch_size)
            ]
            pending = None
            with ThreadPoolExecutor(max_workers=1) as io_executor:
                for i, (start, end) in enumerate(
                    tqdm(batch_ranges, desc='encode_stickers')
                ):
                    ids_batch = list(range(start, end))
                    if pending is not None:
                        tensors_cpu, valid = pending.result()
                    else:
                        tensors_cpu, valid = _materialize_sticker_batch(
                            ids_batch, id2imgpath, args.img_dir, img_process, workers
                        )

                    if i + 1 < len(batch_ranges):
                        ns, ne = batch_ranges[i + 1]
                        pending = io_executor.submit(
                            _materialize_sticker_batch,
                            list(range(ns, ne)),
                            id2imgpath,
                            args.img_dir,
                            img_process,
                            workers,
                        )
                    else:
                        pending = None

                    if device.type == 'cuda':
                        tensors_cpu = tensors_cpu.pin_memory()

                    b = tensors_cpu.size(0)
                    if not valid.any():
                        embs = torch.zeros(b, dim, device=device)
                    elif bool(valid.all()):
                        pv = tensors_cpu.to(device, non_blocking=True)
                        embs = img_clip.model.get_image_features(
                            pixel_values=pv
                        ).float()
                    else:
                        pv = tensors_cpu[valid].to(device, non_blocking=True)
                        ev = img_clip.model.get_image_features(
                            pixel_values=pv
                        ).float()
                        embs = torch.zeros(b, dim, device=device)
                        embs[valid.to(device)] = ev

                    all_embs.append(embs)
        else:
            for start in tqdm(range(0, n_total, batch_size), desc='encode_stickers'):
                end = min(start + batch_size, n_total)
                ids_batch = list(range(start, end))
                img_objs = [None] * len(ids_batch)

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(load_one, i): i for i in ids_batch}
                    for f in as_completed(futures):
                        try:
                            res = f.result()
                            if res is not None:
                                idx, obj = res
                                img_objs[idx - start] = obj
                        except Exception:
                            pass

                valid_indices = [i for i, o in enumerate(img_objs) if o is not None]
                if not valid_indices:
                    dim = (
                        getattr(img_clip, 'get_sentence_embedding_dimension', lambda: 512)()
                        or 512
                    )
                    embs = torch.zeros(len(ids_batch), dim, device=device)
                else:
                    valid_objs = [img_objs[i] for i in valid_indices]
                    img_tokens = img_clip.tokenize(valid_objs)
                    img_tokens = batch_to_device(img_tokens, device)
                    embs_valid = img_clip.forward(img_tokens)['sentence_embedding']
                    embs = torch.zeros(len(ids_batch), embs_valid.size(1), device=device)
                    for j, pos in enumerate(valid_indices):
                        embs[pos] = embs_valid[j]

                all_embs.append(embs)

    out = torch.cat(all_embs, dim=0)
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(out.cpu(), args.output_path)
    print(f'Saved {out.shape} to {args.output_path}')


if __name__ == '__main__':
    main()
