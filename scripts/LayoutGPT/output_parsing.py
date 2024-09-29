import numpy as np
import json
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from matplotlib.patches import Polygon as MplPolygon
import re

# Parse rect data from XML
def parse_rect_data(xml_string):
    rects = []
    pattern = r"<object\s+category='([^']+)'\s+center-x=(\d+\.?\d*)m\s+center-y=(\d+\.?\d*)m\s+center-z=(\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+height=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m\s+orientation=(-?\d+\.?\d*)degrees/>"
    
    matches = re.findall(pattern, xml_string)
    
    for match in matches:
        rect = {
            'category': match[0],
            'x': float(match[1]),
            'y': float(match[2]),
            'z': float(match[3]),
            'width': float(match[4]),
            'height': float(match[5]),
            'depth': float(match[6]),
            'orientation': float(match[7])
        }
        rects.append(rect)
    return rects

# Parse room data from XML
def parse_room_data(xml_string):
    pattern = r"<room\s+category=(\w+)\s+center-x=(-?\d+\.?\d*)m\s+center-z=(-?\d+\.?\d*)m\s+width=(\d+\.?\d*)m\s+depth=(\d+\.?\d*)m/>"
    match = re.search(pattern, xml_string)
    
    if match:
        room = {
            'category': match.group(1),
            'center_x': float(match.group(2)),
            'center_z': float(match.group(3)),
            'width': float(match.group(4)),
            'depth': float(match.group(5))
        }
        return room

def parse_example(example_string):
    # 방 정보 파싱
    room = parse_room_data(example_string)
    
    # 객체 정보 파싱
    objects = parse_rect_data(example_string)
    
    return room, objects

# 색상을 생성하는 함수
def random_color():
    return (random.random(), random.random(), random.random())

# 시각화 함수
def visualize_room_and_objects_shapely(room_data, object_data, scene_id = ""):
    fig, ax = plt.subplots()

    # 방 그리기 (검은색 테두리)
    room_polygon = Polygon([
        (room_data['center_x'] - room_data['width'] / 2, room_data['center_z'] - room_data['depth'] / 2),
        (room_data['center_x'] + room_data['width'] / 2, room_data['center_z'] - room_data['depth'] / 2),
        (room_data['center_x'] + room_data['width'] / 2, room_data['center_z'] + room_data['depth'] / 2),
        (room_data['center_x'] - room_data['width'] / 2, room_data['center_z'] + room_data['depth'] / 2)
    ])
    room_patch = MplPolygon(list(room_polygon.exterior.coords), closed=True, edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(room_patch)
    
    # 오브젝트와 관련된 정보 저장 (범례 표시용)
    legend_patches = []

    # 오브젝트 그리기
    for obj in object_data:
        color = random_color()

        # 오브젝트의 좌표를 생성
        rect = Polygon([
            (-obj['width'] / 2, -obj['depth'] / 2),
            (obj['width'] / 2, -obj['depth'] / 2),
            (obj['width'] / 2, obj['depth'] / 2),
            (-obj['width'] / 2, obj['depth'] / 2)
        ])

        # Shapely로 회전 적용 (degree 단위)
        rotated_rect = rotate(rect, obj['orientation'], use_radians=False)

        # Shapely로 이동 적용 (center를 기준으로 이동)
        moved_rect = translate(rotated_rect, xoff=obj['x'], yoff=obj['z'])

        # Matplotlib을 이용하여 패치로 추가
        object_patch = MplPolygon(list(moved_rect.exterior.coords), closed=True, edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(object_patch)

        # 범례를 위한 패치 추가 (오브젝트 범주와 색상 저장)
        legend_patches.append(plt.Line2D([0], [0], color=color, lw=4, label=obj['category']))

        # 중심에 category 텍스트 표시
        ax.text(
            obj['x'], obj['z'], obj['category'],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8, color='black', fontweight='bold'
        )

    # 축 범위를 설정하고, 축 비율을 동일하게 유지
    ax.set_aspect('equal', 'box')
    plt.xlim(room_data['center_x'] - room_data['width'] / 2 - 1, 
             room_data['center_x'] + room_data['width'] / 2 + 1)
    plt.ylim(room_data['center_z'] - room_data['depth'] / 2 - 1, 
             room_data['center_z'] + room_data['depth'] / 2 + 1)

    # 범례 추가
    ax.legend(handles=legend_patches, loc='upper right', title='Objects')
    plt.title(scene_id)
    # 그래프를 표시하는 대신 저장합니다
    plt.savefig(f'room_layout_{scene_id}.png')
    plt.close()  # 메모리 관리를 위해 figure를 닫습니다

output_file_path = "C:/Users/ttd85/Documents/Resources/IndoorSceneSynthesis/LayoutGPT/inference_results.json"
with open(output_file_path, "r") as file:
    data = json.load(file)
    
example = """
[INST] [INST]
The XZ plane represents the floor, and the Y axis represents the height.
The width refers to the length extending left and right along the X axis.
The height refers to the length extending up and down along the Y axis.
The depth refers to the length extending forward and backward along the Z axis.
Generate a room layout based on the given room requirements.
 [/INST] <room category=bedroom center-x=1.86m center-z=1.11m width=3.72m depth=2.21m/> [/INST]<objects>
 <object category='double_bed' center-x=1.58m center-y=0.43m center-z=1.08m width=1.88m height=0.86m depth=2.15m orientation=0degrees/>
 <object category='pendant_lamp' center-x=1.68m center-y=2.39m center-z=1.16m width=1.01m height=0.89m depth=1.02m orientation=0degrees/>
 <object category='wardrobe' center-x=3.32m center-y=1.21m center-z=0.90m width=1.71m height=2.43m depth=0.60m orientation=-90degrees/>
 <object category='dressing_table' center-x=0.39m center-y=0.71m center-z=0.18m width=1.05m height=1.41m depth=0.60m orientation=0degrees/>
 <object category='armchair' center-x=0.67m center-y=0.46m center-z=1.86m width=0.60m height=0.92m depth=0.57m orientation=90degrees/>
 <object category='desk' center-x=1.35m center-y=0.36m center-z=1.82m width=3.10m height=0.71m depth=0.87m orientation=-180degrees/>
 <object category='floor_lamp' center-x=2.51m center-y=0.79m center-z=2.06m width=0.47m height=1.58m depth=0.49m orientation=19degrees/>
 </objects>
"""

for idx, _data in enumerate(data):
    room, rects = parse_example(_data['output'])
    visualize_room_and_objects_shapely(room, rects, scene_id=f"{idx:03d}")