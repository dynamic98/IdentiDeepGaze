import torch

def positional_encoding(x, L=10):
    frequencies = 2.0 ** torch.arange(L, dtype=torch.float32)
    x_scaled = x.unsqueeze(-1) * frequencies.unsqueeze(0)
    pos_enc = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)
    return pos_enc

def encode_gaze_position(gaze_positions, L=10):
    x_pos_enc = positional_encoding(gaze_positions[..., 0:1], L)
    y_pos_enc = positional_encoding(gaze_positions[..., 1:2], L)
    gaze_pos_enc = torch.cat([x_pos_enc, y_pos_enc], dim=-1)
    return gaze_pos_enc



# 시선 위치 데이터를 정규화하는 함수 정의
def normalize_gaze_positions(gaze_positions, resolution=(1920.0, 1080.0)):
    """
    시선 위치 데이터를 정규화합니다.

    Parameters:
    - gaze_positions: 시선 위치 텐서, shape는 [..., 2]입니다.
    - resolution: 이미지의 해상도를 나타내는 튜플(x_max, y_max)

    Returns:
    - normalized_positions: 정규화된 시선 위치, shape는 [..., 2]입니다.
    """
    x_max, y_max = resolution
    normalized_positions = torch.zeros_like(gaze_positions)
    normalized_positions[..., 0] = (gaze_positions[..., 0] / x_max) * 2 - 1  # x축 정규화
    normalized_positions[..., 1] = (gaze_positions[..., 1] / y_max) * 2 - 1  # y축 정규화
    return normalized_positions


def temporal_positional_encoding(T, L=10):
    """
    시간적 인덱스에 대한 positional encoding을 수행합니다.

    Parameters:
    - T: 프레임의 총 수
    - L: positional encoding의 각 차원에 대한 주파수 수

    Returns:
    - temp_pos_enc: 시간적 인덱스의 positional encoding 결과, shape는 [T, 2*L]
    """
    # 시간적 인덱스 생성
    time_indices = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # [T, 1]
    # Positional encoding 적용
    temp_pos_enc = positional_encoding(time_indices, L)
    return temp_pos_enc

if __name__ == "__main__":

    # 프레임 수
    T = 84
    L = 6  # 주파수 수

    # 예시 시선 위치 데이터
    gaze_positions = torch.tensor([[960.0, 540.0], [480.0, 270.0]])  # FHD 이미지 내 임의의 위치
    L = 6  # 주파수 수

    # 시선 위치에 대한 positional encoding 적용
    gaze_pos_enc = encode_gaze_position(gaze_positions, L)

    print(gaze_pos_enc.shape, gaze_pos_enc)

    # 시간적 positional encoding 적용
    temp_pos_enc = temporal_positional_encoding(T, L)

    print(temp_pos_enc.shape)  # 결과 shape 확인

    # 시선 위치 데이터 정규화
    gaze_positions_normalized = normalize_gaze_positions(gaze_positions)

    # 정규화된 시선 위치에 대한 positional encoding 적용
    gaze_pos_enc_normalized = encode_gaze_position(gaze_positions_normalized, L)

    gaze_positions_normalized, gaze_pos_enc_normalized.shape, gaze_pos_enc_normalized

    print(gaze_pos_enc_normalized)


