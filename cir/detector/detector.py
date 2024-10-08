#
# This file originates from
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/tree/c50df7816714007c7f2f5188995807b3b396ad3d, licensed
# under the MIT license (see CIR-LICENSE in the root folder of this repository).
#
import sys

import gin
import numpy as np
import pandas as pd
import torch

from .tensor_collection import PandasTensorCollection


class Detector:
    def __init__(self, model):
        model.eval()
        self.model = model
        self.config = model.config
        self.category_id_to_label = {v: k for k, v in self.config.label_to_category_id.items()}

    def cast(self, obj):
        return obj.cuda()

    @torch.no_grad()
    @gin.configurable
    def get_detections(self, images, mask_th, detection_th=None,
                       output_masks=True, one_instance_per_class=False):
        images = self.cast(images).float()
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.max() > 1:
            images = images / 255.
            images = images.float().cuda()

        try:
            outputs_ = self.model([image_n for image_n in images])
        except RuntimeError as e:
            print("Detector failed with error:", str(e))
            sys.exit(9)

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n['labels'] = [self.category_id_to_label[category_id.item()] \
                                   for category_id in outputs_n['labels']]
            for obj_id in range(len(outputs_n['boxes'])):
                bbox = outputs_n['boxes'][obj_id]
                info = dict(
                    batch_im_id=n,
                    label=outputs_n['labels'][obj_id],
                    score=outputs_n['scores'][obj_id].item(),
                )
                mask = outputs_n['masks'][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            bboxes = torch.stack(bboxes).cuda().float()
            masks = torch.stack(masks).cuda()
        else:
            infos = dict(score=[], label=[], batch_im_id=[])
            bboxes = torch.empty(0, 4).cuda().float()
            masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool).cuda()

        outputs = PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            outputs.register_tensor('masks', masks)
        if detection_th is not None:
            keep = np.where(outputs.infos['score'] > detection_th)[0]
            outputs = outputs[keep]

        if one_instance_per_class:
            infos = outputs.infos
            infos['det_idx'] = np.arange(len(infos))
            keep_ids = infos.sort_values('score', ascending=False).drop_duplicates('label')['det_idx'].values
            outputs = outputs[keep_ids]
            outputs.infos = outputs.infos.drop('det_idx', axis=1)
        return outputs

    def __call__(self, *args, **kwargs):
        return self.get_detections(*args, **kwargs)
