"""Independent PyTorch/SciPy reference for the explicitly reduced final-head recipe.

No training tensors or user images are uploaded. Run in the isolated detector venv.
The synthetic fixture tests arithmetic only; --pretrained additionally uses the
pinned complete COCO checkpoint and never substitutes a random localization head.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"
CONFIG_SHA256 = "7edf2d96d3deb10aa4ec920be0e8e444b617454545bb58093e32bd8e2b96fa2f"
MODELING_SHA256 = "f0d05693550a30d0fe049d9c5bdd77f4363df5be06c8ab3cf6f7bd07fd6a74f0"
PREFIX = "model.decoder."
TRAINABLE = [f"{PREFIX}bbox_embed.2.layers.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")] + [f"{PREFIX}class_embed.2.{kind}" for kind in ("weight", "bias")]
FROZEN = [f"{PREFIX}lqe_layers.2.reg_conf.layers.{i}.{kind}" for i in range(2) for kind in ("weight", "bias")]


def serial(tensor):
    return {"shape": list(tensor.shape), "data": tensor.detach().cpu().flatten().tolist()}


def linear(x, weights, stem):
    return F.linear(x, weights[stem + ".weight"], weights[stem + ".bias"])


def forward(features, weights, config):
    x = features["bboxInput"]
    for i in range(3):
        x = linear(x, weights, f"{PREFIX}bbox_embed.2.layers.{i}")
        if i < 2:
            x = F.relu(x)
    corners = x + features["residualCorners"]
    probs = corners.reshape(config["queries"], 4, config["bins"]).softmax(-1)
    distances = F.linear(probs, config["projection"])
    refs = features["references"]
    scale = config["regScale"]
    xy1 = refs[:, :2] - (0.5 * scale + distances[:, :2]) * refs[:, 2:] / scale
    xy2 = refs[:, :2] + (0.5 * scale + distances[:, 2:]) * refs[:, 2:] / scale
    boxes = torch.cat([(xy1 + xy2) / 2, xy2 - xy1], -1)
    top = probs.topk(config["topK"], dim=-1).values
    stat = torch.cat([top, top.mean(dim=-1, keepdim=True)], -1).reshape(config["queries"], -1)
    quality = linear(F.relu(linear(stat, weights, f"{PREFIX}lqe_layers.2.reg_conf.layers.0")), weights, f"{PREFIX}lqe_layers.2.reg_conf.layers.1")
    logits = linear(features["classInput"], weights, f"{PREFIX}class_embed.2") + quality
    return {"logits": logits, "boxes": boxes, "corners": corners}


def to_xyxy(boxes):
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], -1)


def ious(boxes1, boxes2):
    a, b = to_xyxy(boxes1), to_xyxy(boxes2)
    low = torch.maximum(a[:, None, :2], b[None, :, :2])
    high = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersect = (high - low).clamp(min=0).prod(-1)
    union = boxes1[:, 2:].prod(-1)[:, None] + boxes2[:, 2:].prod(-1)[None, :] - intersect
    iou = intersect / union
    enclosing = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) - torch.minimum(a[:, None, :2], b[None, :, :2])).prod(-1)
    return iou, iou - (enclosing - union) / enclosing


def loss_and_match(prediction, targets):
    logits, boxes = prediction["logits"], prediction["boxes"]
    labels, target_boxes = targets["labels"], targets["boxes"]
    n = max(len(labels), 1)
    with torch.no_grad():
        prob = logits.sigmoid()[:, labels]
        positive = .25 * (1 - prob) ** 2 * -(prob + 1e-8).log()
        negative = .75 * prob ** 2 * -(1 - prob + 1e-8).log()
        _, all_giou = ious(boxes, target_boxes)
        cost = 2 * (positive - negative) + 5 * torch.cdist(boxes, target_boxes, p=1) - 2 * all_giou
        pi, ti = linear_sum_assignment(cost.numpy())
        target = torch.zeros_like(logits)
        scores = torch.zeros_like(logits)
        if len(pi):
            target[pi, labels[ti]] = 1
            scores[pi, labels[ti]] = ious(boxes[pi], target_boxes[ti])[0].diag()
        weight = .75 * logits.sigmoid().pow(2) * (1 - target) + scores
    vfl = F.binary_cross_entropy_with_logits(logits, scores, weight=weight, reduction="sum") / n
    if len(pi):
        l1 = F.l1_loss(boxes[pi], target_boxes[ti], reduction="sum") / n
        giou = (1 - ious(boxes[pi], target_boxes[ti])[1].diag()).sum() / n
    else:
        l1 = boxes.sum() * 0
        giou = boxes.sum() * 0
    total = vfl + 5 * l1 + 2 * giou
    return {"vfl": vfl, "l1": l1, "giou": giou, "total": total}, {"predictionIndices": pi.tolist(), "targetIndices": ti.tolist()}


def synthetic():
    q, h, c, bins, k, lqe = 4, 3, 2, 5, 2, 4
    config = dict(queries=q, hiddenSize=h, classes=c, bins=bins, regScale=4, topK=k, projection=torch.tensor([-4, -1, 0, 1, 4], dtype=torch.float32))
    def vals(shape, offset, scale=.2):
        count = math.prod(shape)
        return (torch.arange(count, dtype=torch.float32).add(offset).sin().mul(scale)).reshape(shape)
    weights = {}
    for i, dims in enumerate([(h, h), (h, h), (4 * bins, h)]):
        weights[f"{PREFIX}bbox_embed.2.layers.{i}.weight"] = vals(dims, i + 1)
        weights[f"{PREFIX}bbox_embed.2.layers.{i}.bias"] = vals((dims[0],), i + 11).add(.2)
    weights[f"{PREFIX}class_embed.2.weight"] = vals((c, h), 21)
    weights[f"{PREFIX}class_embed.2.bias"] = vals((c,), 31)
    for i, dims in enumerate([(lqe, 4 * (k + 1)), (1, lqe)]):
        weights[f"{PREFIX}lqe_layers.2.reg_conf.layers.{i}.weight"] = vals(dims, i + 41)
        weights[f"{PREFIX}lqe_layers.2.reg_conf.layers.{i}.bias"] = vals((dims[0],), i + 51).add(.2)
    features = dict(bboxInput=vals((q,h), 61, .8), classInput=vals((q,h), 71, .8), residualCorners=vals((q,4*bins), 81, .8), references=torch.tensor([[.2,.2,.2,.2],[.5,.5,.3,.3],[.8,.8,.2,.2],[.3,.7,.2,.25]],dtype=torch.float32))
    return config, features, weights, None


def pretrained(cache):
    from safetensors.torch import load_file
    from transformers import DFineConfig, DFineForObjectDetection
    from transformers.models.d_fine.modeling_d_fine import weighting_function
    import transformers.models.d_fine.modeling_d_fine as modeling
    for name, version in {"torch":"2.6.0+cpu", "numpy":"2.2.3", "scipy":"1.15.2", "transformers":"4.56.0", "safetensors":"0.5.3"}.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Expected {name}=={version} for independent reference generation.")
    source = cache / "model.safetensors"
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned pretrained checkpoint hash mismatch.")
    config_path = cache / "SOURCE_CONFIG.json"
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Pinned pretrained configuration hash mismatch.")
    if hashlib.sha256(Path(modeling.__file__).read_bytes()).hexdigest() != MODELING_SHA256:
        raise ValueError("Pinned Transformers source hash mismatch.")
    conf = DFineConfig.from_json_file(config_path)
    model = DFineForObjectDetection(conf).eval()
    state = load_file(source)
    for key in model.state_dict():
        if key not in state:
            state[key] = state[PREFIX + key]
    model.load_state_dict(state, strict=True)
    captured = {}
    hooks = [
        model.bbox_embed[2].register_forward_pre_hook(lambda _m, xs: captured.update(bboxInput=xs[0])),
        model.class_embed[2].register_forward_pre_hook(lambda _m, xs: captured.update(classInput=xs[0])),
        model.bbox_embed[0].register_forward_hook(lambda _m, _xs, y: captured.update(corners0=y)),
        model.bbox_embed[1].register_forward_hook(lambda _m, _xs, y: captured.update(corners1=y)),
    ]
    y, x = np.mgrid[0:640, 0:640]
    image = np.stack([(7*x+3*y)%256,(2*x+11*y)%256,(13*x+5*y)%256]).astype(np.float32) / 255
    with torch.no_grad():
        output = model(pixel_values=torch.from_numpy(image[None]), return_dict=True)
    for hook in hooks:
        hook.remove()
    features = {"bboxInput":captured["bboxInput"][0], "classInput":captured["classInput"][0], "residualCorners":(captured["corners0"]+captured["corners1"])[0], "references":output.initial_reference_points[0,-1]}
    config = dict(queries=300, hiddenSize=128, classes=80, bins=33, regScale=4, topK=4, projection=weighting_function(32, model.model.decoder.up, model.model.decoder.reg_scale).detach())
    weights = {key:state[key].clone() for key in TRAINABLE+FROZEN}
    prediction = forward(features, weights, config)
    parity = {name:float((prediction[name] - reference[0]).abs().max()) for name,reference in [("logits",output.logits),("boxes",output.pred_boxes)]}
    if any(value > .0002 for value in parity.values()):
        raise ValueError(f"Pretrained head reconstruction failed: {parity}")
    print("Original pretrained head reconstruction:", parity, flush=True)
    return config, features, weights, parity


def generate(real, cache):
    config, features, initial, source_parity = pretrained(cache) if real else synthetic()
    q, c = config["queries"], config["classes"]
    targets = {
        "empty": dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty((0,4),dtype=torch.float32)),
        "single": dict(labels=torch.tensor([0]),boxes=torch.tensor([[.5,.5,.4,.3]],dtype=torch.float32)),
        "multiple": dict(labels=torch.tensor([0,min(c-1,15)]),boxes=torch.tensor([[.2,.3,.25,.35],[.7,.65,.2,.3]],dtype=torch.float32)),
    }
    cases = {}
    for name, target in targets.items():
        weights = {key:value.detach().clone().requires_grad_(key in TRAINABLE) for key,value in initial.items()}
        prediction = forward(features, weights, config)
        losses,matches = loss_and_match(prediction,target)
        losses["total"].backward()
        gradients = {key:serial(weights[key].grad) for key in TRAINABLE}
        case = dict(targets={"labels":target["labels"].tolist(),"boxes":target["boxes"].flatten().tolist()},prediction={key:serial(value) for key,value in prediction.items()},losses={key:float(value.detach()) for key,value in losses.items()},matches=matches,gradients=gradients)
        if name == "multiple":
            steps=[]
            for step in range(2):
                if step:
                    prediction=forward(features,weights,config)
                    losses,matches=loss_and_match(prediction,target)
                    losses["total"].backward()
                with torch.no_grad():
                    for key in TRAINABLE:
                        weights[key].sub_(weights[key].grad,alpha=.0001)
                        weights[key].grad=None
                after=forward(features,weights,config)
                after_loss,after_match=loss_and_match(after,target)
                steps.append(dict(before={key:float(value.detach()) for key,value in losses.items()},after={key:float(value.detach()) for key,value in after_loss.items()},weights={key:serial(weights[key]) for key in TRAINABLE},prediction={key:serial(value) for key,value in after.items()},matches=matches))
            case["steps"]=steps
        cases[name]=case
    source = dict(model="ustc-community/dfine-nano-coco",revision="066438d3d8f0da137a37b38fdf3368fd4afceced",sha256=SOURCE_SHA256,configSha256=CONFIG_SHA256,modelingSha256=MODELING_SHA256) if real else None
    return dict(schemaVersion=1,scope="pretrained-final-head" if real else "synthetic-arithmetic-only",source=source,runtime=dict(torch=torch.__version__,numpy=np.__version__,scipy=importlib.metadata.version("scipy")),sourceParity=source_parity,config={**config,"projection":config["projection"].tolist()},features={key:serial(value) for key,value in features.items()},weights={key:serial(value) for key,value in initial.items()},cases=cases)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pretrained",action="store_true")
    parser.add_argument("--cache-dir",type=Path,default=ROOT / "artifacts/p1b/detector-assets/source",
                        help="Checksum-verified public source cache from prepare_detector.py. No download is performed here.")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=generate(args.pretrained,args.cache_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"bytes":args.output.stat().st_size,"scope":result["scope"],"losses":{name:case["losses"] for name,case in result["cases"].items()}},indent=2))
