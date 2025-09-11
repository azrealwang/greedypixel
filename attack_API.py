from typing import List, Tuple
import os
import argparse
import torch
from itertools import product,cycle
from robustbench import load_model
from utils import load_image_tensor,save_image_tensor
from greedypixel import GreedyPixel

# --------- GOOGLE / AWS / AZURE ---------
from google.cloud import vision
from google.oauth2 import service_account
import boto3
import requests

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    # attack settings
    parser.add_argument("--eps", help="perturbation budget: epsilon/255", type=float, default=255)
    parser.add_argument("--max_query", help="maximum query cost", type=int, default=400)
    parser.add_argument("--surrogate", help="for computing gradient map; none is random", type=str, default=None)
    parser.add_argument("--early_stop", help="stop when label changes", action="store_true")
    # target
    parser.add_argument("--api", help="google, aws, azure", type=str, required=True)
    parser.add_argument("--top_n", help="focus on only top-n predictions", type=int, default=5)
    # input and output
    parser.add_argument("--data", help="only imagenet", type=str, default="imagenet")
    parser.add_argument("--input", help="image file path", type=str, required=True)
    parser.add_argument("--output", help="directory", type=str, required=True)
    args = parser.parse_args()

    return args


# ---------- MultiCloudClassifier ----------
class MultiCloudClassifier:
    """
    classify_one(api_name, image_path, top_n) -> List[Tuple[str, float]]
    GOOGLE_KEY_PATH = "/absolute/path/to/your-service-account.json"
    AWS_ACCESS_KEY = "YOUR_AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "YOUR_AWS_SECRET_KEY"
    AWS_REGION     = "us-east-1"
    AZURE_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com/vision/v3.2/analyze"
    AZURE_KEY      = "YOUR_AZURE_KEY"
    """

    GOOGLE_KEY_PATH = "/absolute/path/to/your-service-account.json"
    AWS_ACCESS_KEY = "YOUR_AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "YOUR_AWS_SECRET_KEY"
    AWS_REGION     = "us-east-1"
    AZURE_ENDPOINT = "https://<your-region>.api.cognitive.microsoft.com/vision/v3.2/analyze"
    AZURE_KEY      = "YOUR_AZURE_KEY"

    def __init__(self):
        gcreds = service_account.Credentials.from_service_account_file(self.GOOGLE_KEY_PATH)
        self._gclient = vision.ImageAnnotatorClient(credentials=gcreds)
        self._aws = boto3.client(
            "rekognition",
            aws_access_key_id=self.AWS_ACCESS_KEY,
            aws_secret_access_key=self.AWS_SECRET_KEY,
            region_name=self.AWS_REGION,
        )
        self._azure_endpoint = self.AZURE_ENDPOINT
        self._azure_key = self.AZURE_KEY

    def classify_one(self, api_name: str, image_path: str, top_n: int = 5) -> List[Tuple[str, float]]:
        api = api_name.lower()
        if api == "google":
            return self._google(image_path, top_n)
        elif api == "aws":
            return self._aws_detect(image_path, top_n)
        elif api == "azure":
            return self._azure(image_path, top_n)
        else:
            raise ValueError("Unsupported API name")

    def _google(self, image_path, top_n):
        with open(image_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        resp = self._gclient.label_detection(image=image)
        anns = sorted(resp.label_annotations, key=lambda a: a.score, reverse=True)
        return [(a.description, float(a.score)) for a in anns[:top_n]]

    def _aws_detect(self, image_path, top_n):
        with open(image_path, "rb") as f:
            data = f.read()
        resp = self._aws.detect_labels(Image={"Bytes": data}, MaxLabels=top_n)
        labels = sorted(resp.get("Labels", []), key=lambda x: x["Confidence"], reverse=True)
        return [(item["Name"], float(item["Confidence"] / 100.0)) for item in labels[:top_n]]

    def _azure(self, image_path, top_n):
        with open(image_path, "rb") as f:
            data = f.read()
        headers = {
            "Ocp-Apim-Subscription-Key": self._azure_key,
            "Content-Type": "application/octet-stream",
        }
        params = {"visualFeatures": "Tags"}
        resp = requests.post(self._azure_endpoint, headers=headers, params=params, data=data, timeout=30)
        resp.raise_for_status()
        tags = sorted(resp.json().get("tags", []), key=lambda x: x["confidence"], reverse=True)
        return [(t["name"], float(t["confidence"])) for t in tags[:top_n]]

def run_attack(args):
    # Initialization
    clf = MultiCloudClassifier()
    x = load_image_tensor(args.input)
    x_orig = x.clone().detach()  # keep original image for L∞ constraint
    eps = args.eps / 255.0  # eps already normalized
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("_tmp", exist_ok=True)
    tmp_path = os.path.join("_tmp", "current.png")

    surrogate = load_model(args.surrogate, dataset=args.data, threat_model="Linf") if args.surrogate else None

    # Compute priority map
    attack = GreedyPixel(clf, surrogate)
    pixel_order = attack.compute_gradient_order(x)

    # Initial API query
    save_image_tensor(x, tmp_path)
    initial_topn = clf.classify_one(args.api, tmp_path, top_n=args.top_n)
    gt_label = initial_topn[0][0]
    best_gt_conf = initial_topn[0][1]
    print(f"\nInitial Top-{args.top_n}:")
    for lbl, conf in initial_topn:
        print(f"  {lbl}: {conf:.4f}")

    query = 0
    label_changed = False

    # Precompute candidate deltas (8 corners in L∞ ball)
    deltas = [torch.tensor(delta) for delta in product([-eps, eps], repeat=3)]

    for it, (r, c) in enumerate(cycle(pixel_order), start=1):
        if query >= args.max_query:
            break

        center_px = x_orig[0, :, r, c]  # always use ORIGINAL pixel as reference
        best_px = x[0, :, r, c].clone()  # fallback to current pixel
        best_conf = best_gt_conf

        for delta in deltas:
            cand = (center_px + delta).clamp(0, 1)
            x[0, :, r, c] = cand

            save_image_tensor(x, tmp_path)
            topn = clf.classify_one(args.api, tmp_path, top_n=args.top_n)
            query += 1

            if not topn:
                print("[WARN] API returned no predictions, skipping candidate.")
                continue

            # Check if GT label still exists in top-n
            gt_conf = None
            for lbl, conf in topn:
                if lbl == gt_label:
                    gt_conf = conf
                    break

            # Success: GT disappeared
            if gt_conf is None:
                save_image_tensor(x, os.path.join(args.output, "success.png"))
                print(f"\nGround-truth '{gt_label}' not found in top-{args.top_n} after {query} queries. "
                      f"Saved image to {args.output}/success.png")
                for lbl, conf in topn:
                    print(f"  {lbl}: {conf:.4f}")
                print(f"[INFO] Attack stopped at pixel #{it} after {query} queries.")
                return

            # Top-1 label changed
            if topn[0][0] != gt_label and not label_changed:
                label_changed = True
                save_image_tensor(x, os.path.join(args.output, "label_change.png"))
                print(f"\nTop-1 changed from '{gt_label}' to '{topn[0][0]}' after {query} queries. "
                      f"Saved image to {args.output}/label_change.png")
                for lbl, conf in topn:
                    print(f"  {lbl}: {conf:.4f}")
                if args.early_stop:
                    print(f"[INFO] Attack stopped at pixel #{it} after {query} queries.")
                    return

            # Pick candidate minimizing GT confidence
            if gt_conf < best_conf:
                best_conf = gt_conf
                best_px = cand

            if query >= args.max_query:
                break

        # Set best pixel found and project back into L∞ ball
        x[0, :, r, c] = best_px
        x = torch.max(torch.min(x, x_orig + eps), x_orig - eps).clamp(0, 1)

    print(f"\nReached {args.max_query} max queries.")
    save_image_tensor(x, os.path.join(args.output, "max_query.png"))
    final_topn = clf.classify_one(args.api, os.path.join(args.output, "max_query.png"), top_n=args.top_n)
    for lbl, conf in final_topn:
        print(f"  {lbl}: {conf:.4f}")

# ---------- Main ----------
if __name__ == "__main__":
    args = parse_args_and_config()
    print(args)
    run_attack(args)
