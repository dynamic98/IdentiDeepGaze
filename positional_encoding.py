import torch
import torch.nn as nn
import math

class PositionalEncoding_NeRF(nn.Module):
    """
    Neural Radiance Fields(NeRF)를 위한 위치 인코딩 클래스입니다.
    이 인코딩은 입력 좌표에 대한 고주파 정보를 모델에 제공하여 더 세밀한 디테일을 생성할 수 있게 합니다.
    
    Args:
        L (int): 생성할 주파수의 개수입니다.
        max_L (int): 최대 로그 스케일의 주파수입니다. 이 값은 주파수의 분포를 결정합니다.
        include_input (bool): 인코딩에 입력값을 포함시킬지 여부를 결정합니다.
    """
    def __init__(self, L=10, max_L=5, include_input=True):
        super(PositionalEncoding_NeRF, self).__init__()
        self.include_input = include_input
        # 주파수를 로그 스케일로 생성합니다.
        freqs = 2.0 ** torch.linspace(0, max_L, steps=L, dtype=torch.float32)
        # 생성된 주파수를 모듈의 버퍼로 등록합니다. 이는 학습 중에는 변하지 않지만, 모델의 상태에 포함되어야 하는 값들입니다.
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        입력된 x에 대한 위치 인코딩을 수행합니다.
        
        Args:
            x (Tensor): 입력 텐서입니다.
            
        Returns:
            Tensor: 위치 인코딩이 적용된 텐서입니다.
        """
        # 입력값의 마지막 차원을 확장합니다.
        x = x.unsqueeze(-1)
        # 사인과 코사인 함수를 사용하여 주파수에 따른 인코딩을 수행합니다.
        sin_enc = torch.sin(x * self.freqs)
        cos_enc = torch.cos(x * self.freqs)
        # 사인과 코사인 인코딩을 결합하고, 마지막 두 차원을 평탄화합니다.
        enc = torch.cat([sin_enc, cos_enc], dim=-1).flatten(start_dim=-2)
        # include_input이 True인 경우, 원본 입력값을 인코딩에 추가합니다.
        if self.include_input:
            enc = torch.cat([x.squeeze(-1), enc], dim=-1)
        return enc

class PositionalEncoding(nn.Module):
    """
    트랜스포머 모델을 위한 위치 인코딩 클래스입니다.
    이 인코딩은 시퀀스 내 각 위치에 대한 고유한 인코딩을 제공하여, 모델이 위치 정보를 활용할 수 있게 합니다.
    
    Args:
        d_model (int): 인코딩의 차원입니다. 모델의 입력과 일치해야 합니다.
        max_len (int): 인코딩을 생성할 최대 길이입니다. 이 값은 데이터셋의 최대 시퀀스 길이를 고려하여 설정합니다.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # max_len 길이와 d_model 차원을 가지는 텐서를 0으로 초기화합니다.
        self.encoding = torch.zeros(max_len, d_model)
        # 각 위치에 대한 인덱스를 생성합니다.
        position = torch.arange(0, max_len).unsqueeze(1)
        # 디모델 차원에 대한 분할 항을 계산합니다. 이는 주파수의 변화를 결정합니다.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 사인과 코사인 함수를 사용하여 각 위치에 대한 인코딩을 계산합니다.
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        # 배치 차원을 추가합니다.
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        입력된 x에 대한 위치 인코딩을 반환합니다.
        
        Args:
            x (Tensor): 입력 텐서입니다.
            
        Returns:
            Tensor: 위치 인코딩이 추가된 텐서입니다. 이는 입력 텐서 x의 길이에 따라 조정됩니다.
        """
        # x의 길이에 맞게 인코딩을 잘라내고, 배치 차원을 제거합니다.
        return self.encoding[:, :x.size(1)].squeeze(0)



if __name__ == "__main__":

    # 프레임 수
    T = 84
    L = 6  # 주파수 수

    # 예시 시선 위치 데이터
    gaze_positions = torch.tensor([[960.0, 540.0], [480.0, 270.0]])  # FHD 이미지 내 임의의 위치
    pos_encoder_nerf = PositionalEncoding_NeRF(L=7)
    print(pos_encoder_nerf(gaze_positions).shape)

    pos_encoder = PositionalEncoding(d_model=30)
    print(pos_encoder(gaze_positions).shape)




