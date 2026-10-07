from __future__ import annotations
import torch

def resolve_device(requested=None, configured='auto'):
    """Resolve auto/cpu/cuda/cuda:index; explicit unavailable CUDA is an error."""
    value = requested or configured or 'auto'
    if value == 'auto':
        return ('cuda:0' if torch.cuda.is_available() else 'cpu'), False
    if value == 'cpu':
        return 'cpu', False
    if value == 'cuda':
        value = 'cuda:0'
    if value.startswith('cuda:'):
        try: index = int(value.split(':', 1)[1])
        except ValueError as e: raise ValueError(f'잘못된 device 값: {value}') from e
        if index < 0: raise ValueError(f'잘못된 CUDA 인덱스: {index}')
        if not torch.cuda.is_available(): raise RuntimeError(f'{value} 요청됐지만 CUDA를 사용할 수 없습니다. device=auto 또는 cpu를 지정하세요.')
        if index >= torch.cuda.device_count(): raise RuntimeError(f'{value} 요청됐지만 CUDA 장치 수는 {torch.cuda.device_count()}개입니다.')
        return f'cuda:{index}', False
    raise ValueError(f"지원하지 않는 device '{value}'. auto, cpu, cuda, cuda:index 중 하나를 사용하세요.")

def environment_info(device):
    info = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device': device}
    if device.startswith('cuda') and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        info.update(cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(idx), gpu_index=idx)
    return info

def format_environment(info):
    return 'Environment: ' + ', '.join(f'{k}={v}' for k,v in info.items())
