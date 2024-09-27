import os
import sys
from tqdm import tqdm

from object_utils import Scene

def list_all_file_paths(directory):
    """
    지정된 디렉토리 내의 모든 파일 경로를 반환합니다.
    
    Args:
        directory (str): 검색할 디렉토리의 경로.
        
    Returns:
        list: 모든 파일의 절대 경로가 담긴 리스트.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

# 사용 예시
if __name__ == "__main__":
    directory = '../../Resources/IndoorSceneSynthesis/3D-FRONT/'  # 여기에 검색하려는 폴더의 경로를 입력하세요.
    all_files = list_all_file_paths(directory)
    for file in tqdm(all_files):
        scene = Scene(file)

        for room in scene.rooms:
            print(room)

            for furniture in room.children:
                print(furniture)

        exit()