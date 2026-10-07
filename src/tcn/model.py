"""
Skeleton-TCN 모델 정의 — 원조 ST-GCN(Yan et al., 2018)에서 공간 그래프 합성곱(GCN)만 제거한 구조

원조 ST-GCN 각 유닛 = [공간 그래프 합성곱(GCN)] → [시간 합성곱(TCN)]
이 파일 = 위에서 GCN 을 빼고 TCN 만 남김. 나머지(채널·블록수·stride·residual·dropout)는 원조와 동일.

원조 사양 (Yan 2018):
  - 9개 유닛, 채널 64/64/64/128/128/128/256/256/256
  - 시간 커널 9, 4번째·7번째 유닛 stride=2 (시간축 풀링)
  - 각 유닛 residual + dropout(0.5)
  - global average pooling → 256차원 → softmax

핵심: 관절 차원 V(=17)를 끝까지 유지. 시간 합성곱은 Conv2d 로 (시간축 9 × 관절축 1) 만 →
      관절끼리 안 섞음(그게 원래 GCN 담당이었으므로, GCN 을 뺀 지금은 관절 독립 처리).
"""

import torch
import torch.nn as nn


class TemporalUnit(nn.Module):
    """
    원조 ST-GCN 유닛에서 GCN 을 뺀 '시간 합성곱만' 유닛.
    입력·출력 : (N, C, T, V)   ← 관절 차원 V 유지
    시간 합성곱 : Conv2d(kernel=(9,1))  → 시간축 9프레임, 관절축은 1 (관절끼리 안 섞음)
    """
    def __init__(self, in_ch, out_ch, stride=1, t_kernel=9, dropout=0.5):
        super().__init__()
        pad = (t_kernel - 1) // 2   # 시간축 길이 유지 (stride=1일 때)
        self.tcn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(t_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # residual (원조와 동일: 채널·stride 안 맞으면 1x1 conv 로 맞춤)
        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        out = self.tcn(x)
        return self.relu(out + res)


class SkeletonTCN(nn.Module):
    """
    원조 ST-GCN 에서 GCN 만 제거한 Skeleton-TCN.
    입력 : (N, C=3, T, V=17)   = 배치 × 좌표 × 프레임 × 관절
    출력 : (N, num_classes) logits
    특징 벡터(256) 만 뽑고 싶으면 extract_features() 사용.
    """
    # (in_ch, out_ch, stride) × 9  — 원조 ST-GCN 과 동일 (4·7번째 stride=2)
    UNIT_CFG = [
        (3,   64,  1),
        (64,  64,  1),
        (64,  64,  1),
        (64,  128, 2),   # 4번째: 시간축 풀링
        (128, 128, 1),
        (128, 128, 1),
        (128, 256, 2),   # 7번째: 시간축 풀링
        (256, 256, 1),
        (256, 256, 1),
    ]

    def __init__(self, in_channels=3, num_joints=17, num_classes=2, t_kernel=9):
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints

        # 원조와 동일: 입력 정규화 (관절·좌표를 펼쳐 BN)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # (in_ch, out_ch, stride) × 9 — 첫 유닛만 in_channels 로, 나머지는 원조 고정
        unit_cfg = [
            (in_channels, 64,  1),   # ← in_channels 반영 (3이든 4든)
            (64,  64,  1),
            (64,  64,  1),
            (64,  128, 2),
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 256, 1),
        ]

        self.units = nn.ModuleList(
            TemporalUnit(i, o, stride=s, t_kernel=t_kernel)
            for (i, o, s) in unit_cfg
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _forward_features(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        # 원조 방식 입력 정규화: (N, C*V, T) 로 펼쳐 BN 후 되돌림
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()   # (N, C, T, V) 복원
        for unit in self.units:
            x = unit(x)                              # 관절 차원 V 유지된 채 통과
        x = self.pool(x).view(N, -1)                 # (N, 256)
        return x

    def extract_features(self, x):
        """분류기 통과 전 256차원 특징만 반환 (요약 추출용)."""
        return self._forward_features(x)

    def forward(self, x):
        x = self._forward_features(x)                # (N, 256)
        return self.fc(x)                            # (N, num_classes)


if __name__ == "__main__":
    # 셰이프 자가 점검
    model = SkeletonTCN()
    for T in (16, 64):
        dummy = torch.randn(2, 3, T, 17)
        out = model(dummy)
        feat = model.extract_features(dummy)
        print(f"T={T}: 입력 {tuple(dummy.shape)} → 특징 {tuple(feat.shape)} → 출력 {tuple(out.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params:,}")
