import torch
import torch.nn.functional as F
from detectron2.utils import comm
import open_clip
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.MARIS.CLIP_MODEL_NAME
        pretrained = cfg.MODEL.MARIS.CLIP_PRETRAINED_WEIGHTS
        if comm.get_local_rank() == 0:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained if pretrained else None)
            comm.synchronize()

        self.clip_model = model
        self.model_name = model_name
        self._size_divisibility = 32
        try:
            self.dim_latent = model.text.text_projection.shape[1]
        except Exception:
            self.dim_latent = cfg.MODEL.MARIS.EMBED_DIM

    def forward(self, x):
        visual = self.clip_model.visual
        if hasattr(visual, "trunk"):
            x = visual.trunk.stem(x)
            for stage in visual.trunk.stages:
                x = stage(x)
            dense = visual.trunk.norm_pre(x)
        elif hasattr(visual, "conv1"):
            x = visual.conv1(x)
            x = visual.bn1(x)
            x = visual.relu1(x)
            x = visual.maxpool(x)
            x = visual.layer1(x)
            x = visual.layer2(x)
            x = visual.layer3(x)
            x = visual.layer4(x)
            dense = x
        else:
            dense = visual(x)
        return {"clip_vis_dense": dense}

    def get_text_classifier(self, class_names, device):
        text = open_clip.tokenize(class_names).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def visual_prediction_forward(self, pooled_feature):
        return self.clip_model.visual.head(pooled_feature)

    def encode_image(self, x):
        return self.clip_model.encode_image(x)

    def output_shape(self):
        return {"res5": ShapeSpec(channels=self.dim_latent, stride=32)}

    @property
    def size_divisibility(self):
        return self._size_divisibility
