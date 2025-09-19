import json
import os
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class InputStream:
    def __init__(self, data: str):
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data: List[int], num: int) -> int:
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data: List[int]) -> str:
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.ndarray:
    """Decode Label Studio brush RLE into a binary mask"""
    rle_input = InputStream(bytes2bit(rle))
    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)

    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image


def ls_json_to_instance_masks(json_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for image in data:
        task_id = image.get("id")
        anns = image.get("annotations", [])
        if not anns:
            continue

        results = anns[0].get("result", [])
        if not results:
            continue

        # infer dimensions from first result
        width = results[0]["original_width"]
        height = results[0]["original_height"]

        # instance mask
        mask = np.zeros((height, width), dtype=np.uint16)
        instance_id = 1

        for r in results:
            if r.get("type", "").lower().startswith("brush"):
                rle = r["value"]["rle"]
                inst_mask = rle_to_mask(rle, height, width)
                if np.any(inst_mask):
                    mask[inst_mask > 0] = instance_id
                    instance_id += 1
                else:
                    print(f"Task {task_id}: empty mask for result {r}")

        plt.imshow(mask)
        plt.show()

        filename = image.get("file_upload")
        out_path = os.path.join(output_dir, filename)
        Image.fromarray(mask, mode="I;16").save(out_path)
        print(f"Task {task_id}: {instance_id-1} instances → {out_path}")



if __name__ == "__main__":
    json_file = "labeling/synmt.json"  # path to Label Studio JSON export
    output_dir = "labeling/masks"  # output folder for PNG instance masks

    ls_json_to_instance_masks(json_file, output_dir)
