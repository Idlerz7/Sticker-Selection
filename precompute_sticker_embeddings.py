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

    try:
        img_clip = SentenceTransformer(args.img_pretrain_path)
    except Exception as e:
        print(f"SentenceTransformer failed: {e}, trying HF CLIP...")
        from main import HFClipSentenceEncoder
        img_clip = HFClipSentenceEncoder(args.img_pretrain_path, local_files_only=args.local_files_only)

    if hasattr(img_clip, 'preprocess'):
        img_process = img_clip.preprocess
    else:
        img_process = img_clip._first_module().preprocess

    device = img_clip._target_device
    img_clip.eval()

    all_embs = []
    n_total = args.max_image_id
    batch_size = args.batch_size
    workers = args.workers

    def load_one(i):
        return load_image(i, id2imgpath, args.img_dir, img_process)

    with torch.no_grad():
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
                    except Exception as ex:
                        pass  # skip failed loads

            # Filter out None (missing/failed images)
            valid_indices = [i for i, o in enumerate(img_objs) if o is not None]
            if not valid_indices:
                dim = getattr(img_clip, 'get_sentence_embedding_dimension', lambda: 512)() or 512
                embs = torch.zeros(len(ids_batch), dim, device=device)
            else:
                valid_objs = [img_objs[i] for i in valid_indices]
                img_tokens = img_clip.tokenize(valid_objs)
                img_tokens = batch_to_device(img_tokens, device)
                embs_valid = img_clip.forward(img_tokens)['sentence_embedding']
                # Place back into full batch (zeros for failed)
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
