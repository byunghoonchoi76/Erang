import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms

# ============================================================
# 1. 해상도 및 클래스 인덱스 매핑 정의
# ============================================================
RESOLUTION_MAP = {
    '0.5M': 0,
    '1M': 1,
    '2M': 2,
    '3M': 3,
    '8M': 4
}
IDX_TO_RES = {v: k for k, v in RESOLUTION_MAP.items()}

# ============================================================
# 2. DINOv3 전용 이미지 전처리 트랜스폼 헬퍼
# ============================================================
def get_dinov3_transform(img_size=224):
    """
    DINOv3 모델 입력 규격에 맞춘 전처리 Pipeline을 반환합니다.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================
# 3. DINOv3 해상도 분류 모델 클래스
# ============================================================
class DINOv3ResolutionClassifier(nn.Module):
    def __init__(self, model_name='vit_large_patch16_dinov3', num_classes=5, pretrained=False):
        super().__init__()
        # timm 라이브러리를 활용하여 DINOv3 백본에 5개 해상도 분류용 Head 결합
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# ============================================================
# 4. 가중치 로드 및 모델 초기화 통합 함수
# ============================================================
def load_dinov3_model(checkpoint_path, model_name='vit_large_patch16_dinov3', device='cuda'):
    """
    지정된 체크포인트 가중치를 안전하게 로드하여 추론 모드의 DINOv3 모델을 반환합니다.
    """
    model = DINOv3ResolutionClassifier(model_name=model_name, num_classes=5, pretrained=False)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 딕셔너리 포장 형태(상태 파일) 혹은 순수 state_dict 대응
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 💡 [핵심 수정] 가중치 키(Key) 불일치 자동 맵핑 보정
        fixed_state_dict = {}
        for k, v in state_dict.items():
            # 저장된 체크포인트 키에 'model.' 접두사가 없다면 강제로 붙여서 아키텍처와 짝을 맞춤
            if not k.startswith('model.'):
                fixed_state_dict[f'model.{k}'] = v
            else:
                fixed_state_dict[k] = v

        # 수정한 state_dict로 로드
        model.load_state_dict(fixed_state_dict, strict=True)
        print(f"✅ [DINOv3] 가중치 로드 성공: {os.path.basename(checkpoint_path)}")
    else:
        print(f"⚠️ [DINOv3] 경고: 가중치 파일을 찾을 수 없어 빈 구조만 초기화합니다. 경로: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model
