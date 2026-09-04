"""CUT + Structural Consistency Loss (제안서 핵심 기여).

CUTModel을 상속해 compute_G_loss에 L_SC를 추가한다:
  L_total = L_gen(CUT) + lambda_SC * L_SC
L_SC = 입력 SR_Output(real_A)과 생성 결과(fake_B)의 구조 피처 일관성 (VGG, DINOv3 대기).
--model cutsc --lambda_SC <값> 로 사용.
"""
import os
import sys

from .cut_model import CUTModel


class CUTSCModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_SC', type=float, default=1.0,
                            help='structural consistency loss 가중치 (L_total = L_gen + lambda_SC*L_SC)')
        parser.add_argument('--sc_backend', type=str, default='vgg',
                            help='SC 피처 백엔드: vgg | dinov3')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_SC = 0.0
        if self.isTrain and opt.lambda_SC > 0.0:
            self.loss_names = self.loss_names + ['SC']
            ROOT = os.path.expanduser("~/geoit_sr_refiner")
            if ROOT not in sys.path:
                sys.path.append(ROOT)
            from src.sc_loss import StructuralConsistencyLoss
            dino = None
            if opt.sc_backend == "dinov3":
                # DINOv3 백본 직접 로드 (team_aiduo 미의존, sys.path 오염 방지)
                import timm, yaml, torch
                cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/paths.yaml"), encoding="utf-8"))
                wp = os.path.join(ROOT, cfg["weights"]["dinov3"])
                bb = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=5)
                ck = torch.load(wp, map_location=self.device)
                sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
                bb.load_state_dict(sd, strict=False)
                bb.to(self.device).eval()
                for p in bb.parameters():
                    p.requires_grad_(False)
                dino = type("_DinoWrap", (), {"model": bb})()   # sc_loss가 .model 접근
                print(f"[cutsc] DINOv3 SC 백엔드 로드: {os.path.basename(wp)}")
            self.sc_criterion = StructuralConsistencyLoss(
                backend=opt.sc_backend, dinov3_model=dino, device=self.device)

    def compute_G_loss(self):
        loss_G = super().compute_G_loss()          # CUT의 G_GAN + NCE
        if self.isTrain and self.opt.lambda_SC > 0.0:
            a = (self.real_A + 1) / 2               # [-1,1] -> [0,1]
            b = (self.fake_B + 1) / 2
            self.loss_SC = self.sc_criterion(a, b) * self.opt.lambda_SC
            self.loss_G = loss_G + self.loss_SC
        return self.loss_G
