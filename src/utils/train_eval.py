import math
from typing import Tuple

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_one_epoch(model_obj: torch.nn.Module,
                       loader: DataLoader,
                       accelerator,
                       mode: str = "ar_flow",
                       eval_no_rgb_noise: bool = True) -> Tuple[float, float, float, float]:
    """Unified evaluation loop.

    Args:
        model_obj: JetFormerTrain (ar_flow) or FlowTrain/FlowCore (flow)
        loader: dataloader
        accelerator: accelerator helper with wrap_dataloader/autocast if available
        mode: "ar_flow" for JetFormer pipeline; "flow" for flow-only models
    Returns:
        (total, text_ce_or_0, image_bpd_total_or_flow_bpd, flow_bpd_or_0)
    """
    kind = str(mode).lower()
    model_obj.eval()
    sum_total = 0.0
    sum_text = 0.0
    sum_img = 0.0
    sum_flow = 0.0
    count = 0
    iterable = accelerator.wrap_dataloader(loader, is_train=False) if hasattr(accelerator, 'wrap_dataloader') else loader
    for batch in iterable:
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        bsz = None
        if kind == "ar_flow":
            # Signal to disable RGB noise during eval for reproducibility
            try:
                batch = dict(batch)
                batch['no_rgb_noise'] = bool(eval_no_rgb_noise)
            except Exception:
                pass
            out = model_obj(batch)
            bsz = batch['image'].size(0)
            sum_total += float(out.get('loss', 0.0)) * bsz
            sum_text += float(out.get('text_loss', 0.0)) * bsz
            sum_img += float(out.get('image_bpd_total', out.get('bpd', 0.0))) * bsz
            sum_flow += float(out.get('flow_bpd_component', out.get('logdet_bpd', 0.0))) * bsz
        else:  # flow-only
            images_uint8 = batch["image"]
            bsz = images_uint8.size(0)
            out = model_obj(images_uint8)
            sum_total += float(out.get('bpd', out.get('loss', 0.0))) * bsz
            sum_img += float(out.get('bpd', 0.0)) * bsz
            sum_flow += float(out.get('logdet_bpd', 0.0)) * bsz
        count += int(bsz or 0)
    model_obj.train()
    denom = max(1, count)
    return (sum_total/denom, sum_text/denom, sum_img/denom, sum_flow/denom)


