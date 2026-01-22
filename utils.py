from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def concat_videos_hori(pil_video_lst):
    """
    pil_video_lst의 각 인덱스에 있는 이미지들을 가로로 이어붙여 
    새로운 이미지를 만들고 이를 리스트에 저장하여 반환하는 함수.
    
    Args:
    - pil_video_lst: PIL 이미지 객체들이 담긴 리스트들의 리스트
    
    Returns:
    - 새로운 PIL 이미지들을 담은 리스트
    """
    concatenated_images = []

    # 각 인덱스에 있는 이미지들을 가로로 이어붙임
    for i in range(len(pil_video_lst[0])):  # 첫 번째 그룹의 이미지 개수를 기준으로 순회
        # 현재 인덱스에 해당하는 이미지를 가로로 이어붙이기
        total_width = sum(img.width for img in [pil_video_lst[j][i] for j in range(len(pil_video_lst))])
        max_height = max(img.height for img in [pil_video_lst[j][i] for j in range(len(pil_video_lst))])

        new_img = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for j in range(len(pil_video_lst)):
            img = pil_video_lst[j][i]
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        
        concatenated_images.append(new_img)

    return concatenated_images
