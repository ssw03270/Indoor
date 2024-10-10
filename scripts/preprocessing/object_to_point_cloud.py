import trimesh
import numpy as np

def obj_to_point_cloud(obj_file_path, num_points=10000):
    # obj 파일 로드
    mesh = trimesh.load(obj_file_path)
    
    # 메시에서 포인트 샘플링
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # 포인트 클라우드 반환
    return points

def save_point_cloud(points, output_file_path):
    # 포인트 클라우드를 파일로 저장 (예: .ply 형식)
    pc = trimesh.PointCloud(points)
    pc.export(output_file_path)

if __name__ == "__main__":
    obj_file = "E:/Resources/IndoorSceneSynthesis/InstructScene/3D-FRONT/3D-FUTURE-model/000ffb60-6414-41b2-80cc-e38879db8fe8.raw_model.obj"
    output_file = "path/to/your/output.ply"
    
    # obj 파일을 포인트 클라우드로 변환
    point_cloud = obj_to_point_cloud(obj_file)
    
    # 포인트 클라우드 저장
    save_point_cloud(point_cloud, output_file)
    
    print(f"포인트 클라우드가 {output_file}에 저장되었습니다.")
