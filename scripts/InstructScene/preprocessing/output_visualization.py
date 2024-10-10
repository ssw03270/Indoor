import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors

def parse_rect_data(xml_string):
    root = ET.fromstring(xml_string)
    rects = []
    for rect in root.findall('rect'):
        rects.append({
            'category': rect.get('object-category'),
            'caption': rect.get('object-caption'),
            'x': float(rect.get('x')),
            'y': float(rect.get('y')),
            'width': float(rect.get('width')),
            'height': float(rect.get('height')),
            'direction': float(rect.get('direction')),
            'margin-top': rect.get('margin-top'),
            'margin-right': rect.get('margin-right'),
            'margin-bottom': rect.get('margin-bottom'),
            'margin-left': rect.get('margin-left')
        })
    return rects

def visualize_layout(rects, ax, title):
    colors = list(mcolors.TABLEAU_COLORS.values())
    category_color = {}
    
    for i, rect in enumerate(rects):
        x = rect['x'] - rect['width'] / 2
        y = rect['y'] - rect['height'] / 2
        
        if rect['category'] not in category_color:
            category_color[rect['category']] = colors[len(category_color) % len(colors)]
        
        color = category_color[rect['category']]
        
        rect_patch = Rectangle((x, y), rect['width'], rect['height'], 
                               fill=True, facecolor=color, edgecolor='black', alpha=0.5)
        
        ax.add_patch(rect_patch)
        
        margin_text = f"T:{rect['margin-top']} R:{rect['margin-right']}\nB:{rect['margin-bottom']} L:{rect['margin-left']}"
        ax.text(rect['x'], rect['y'] + rect['height']/2, margin_text, 
                ha='center', va='bottom', fontsize=4, color='red')

    ax.set_aspect('equal')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title(title)
    
    # 범례 추가
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', alpha=0.5) 
                       for color in category_color.values()]
    ax.legend(legend_elements, category_color.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

def visualize_both_layouts(gt_code, output_code):
    gt_rects = parse_rect_data(gt_code)
    output_rects = parse_rect_data(output_code)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    
    # 모든 사각형의 x, y 좌표와 크기를 수집
    all_x = [r['x'] for r in gt_rects + output_rects]
    all_y = [r['y'] for r in gt_rects + output_rects]
    all_widths = [r['width'] for r in gt_rects + output_rects]
    all_heights = [r['height'] for r in gt_rects + output_rects]
    
    # 전체 범위 계산
    max_dim = max(max(all_widths), max(all_heights))
    x_min, x_max = min(all_x) - max_dim, max(all_x) + max_dim
    y_min, y_max = min(all_y) - max_dim, max(all_y) + max_dim
    
    # 두 축에 동일한 한계 설정
    visualize_layout(gt_rects, ax1, 'GT layout')
    visualize_layout(output_rects, ax2, 'Llama layout')
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

# 사용 예시
if __name__ == "__main__":
    gt_code = """
<rects>
  <rect object-category='Wardrobe' object-caption='a wooden wardrobe with a curtain' x='0.11' y='-1.43' width='2.7' height='0.49' direction='0.0' margin-top='proximal'  margin-right='adjacent'  margin-bottom='adjacent'  margin-left='proximal' />
  <rect object-category='King-size Bed' object-caption='a brown and black double bed with a comforter' x='0.44' y='0.22' width='2.1' height='2.09' direction='270.0' margin-top='proximal'  margin-right='adjacent'  margin-bottom='proximal'  margin-left='proximal' />
  <rect object-category='Corner/Side Table' object-caption='a black corner side table with tray' x='0.0' y='1.53' width='0.32' height='0.42' direction='180.0' margin-top='adjacent'  margin-right='proximal'  margin-bottom='adjacent'  margin-left='proximal' />
  <rect object-category='Drawer Chest / Corner cabinet' object-caption='a gray metal cabinet with a metal base' x='-1.25' y='0.45' width='0.51' height='0.69' direction='90.0' margin-top='proximal'  margin-right='proximal'  margin-bottom='distant'  margin-left='adjacent' />
  <rect object-category='Nightstand' object-caption='a nightstand with two drawers' x='1.39' y='-0.74' width='0.39' height='0.48' direction='270.0' margin-top='adjacent'  margin-right='adjacent'  margin-bottom='adjacent'  margin-left='adjacent' />
  <rect object-category='Ceiling Lamp' object-caption='a silver circular ceiling lamp' x='-0.01' y='0.09' width='0.62' height='0.62' direction='90.0' margin-top='distant'  margin-right='distant'  margin-bottom='distant'  margin-left='distant' />
</rects>
    """

    output_code = """
<rects>
  <rect object-category='Wardrobe' object-caption='a wooden wardrobe with a curtain' x='-0.44' y='1.26' width='1.26' height='0.48' direction='180.0' margin-top='adjacent'  margin-right='proximal'  margin-bottom='distant'  margin-left='adjacent' />
  <rect object-category='King-size Bed' object-caption='a brown and black double bed with a comforter' x='-0.07' y='-0.17' width='2.2' height='2.02' direction='90.0' margin-top='proximal'  margin-right='distant'  margin-bottom='distant'  margin-left='adjacent' />
  <rect object-category='Corner/Side Table' object-caption='a black corner side table with tray' x='-1.25' y='-1.32' width='0.35' height='0.42' direction='90.0' margin-top='adjacent'  margin-right='distant'  margin-bottom='adjacent'  margin-left='adjacent' />
  <rect object-category='Drawer Chest / Corner cabinet' object-caption='a gray metal cabinet with a metal base' x='-1.35' y='0.92' width='0.42' height='0.5' direction='90.0' margin-top='adjacent'  margin-right='distant'  margin-bottom='adjacent'  margin-left='adjacent' />
  <rect object-category='Nightstand' object-caption='a nightstand with two drawers' x='-1.24' y='1.1' width='0.56' height='0.55' direction='90.0' margin-top='proximal'  margin-right='distant'  margin-bottom='adjacent'  margin-left='adjacent' />
  <rect object-category='Nightstand' object-caption='a nightstand with two drawers' x='-1.1' y='-1.32' width='0.56' height='0.55' direction='0.0' margin-top='adjacent'  margin-right='distant'  margin-bottom='adjacent'  margin-left='proximal' />
  <rect object-category='Ceiling Lamp' object-caption='a silver circular ceiling lamp' x='-0.04' y='-0.03' width='0.65' height='0.66' direction='90.0' margin-top='distant'  margin-right='distant'  margin-bottom='distant'  margin-left='distant' />
</rects>
    """

    output_code = """
<rects>
  <rect object-category='King-size Bed' object-caption='a yellow and gray double bed with pillows and a blanket' x='-1.07' y='-0.38' width='2.15' height='1.85' direction='0.0' margin-top='proximal'  margin-right='distant'  margin-bottom='distant'  margin-left='distant' />
  <rect object-category='Nightstand' object-caption='a nightstand with a magazine and books' x='-2.17' y='1.06' width='0.31' height='0.55' direction='90.0' margin-top='distant'  margin-right='proximal'  margin-bottom='distant'  margin-left='distant' />
  <rect object-category='Nightstand' object-caption='a nightstand with a magazine and books' x='-2.19' y='-1.48' width='0.31' height='0.55' direction='90.0' margin-top='distant'  margin-right='adjacent'  margin-bottom='distant'  margin-left='proximal' />
  <rect object-category='Drawer Chest / Corner cabinet' object-caption='a wooden cabinet' x='-1.28' y='1.81' width='0.31' height='0.55' direction='90.0' margin-top='proximal'  margin-right='proximal'  margin-bottom='proximal'  margin-left='distant' />
  <rect object-category='Pendant Lamp' object-caption='a black rectangular pendant lamp with two wires' x='-0.01' y='0.14' width='1.0' height='0.3' direction='180.0' margin-top='distant'  margin-right='distant'  margin-bottom='distant'  margin-left='distant' />
</rects>
    """
    visualize_both_layouts(gt_code, output_code)