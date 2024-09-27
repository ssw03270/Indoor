from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf, UsdShade

def create_living_room_scene(output_file):
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(output_file)
    
    # Set the default prim
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/LivingRoom').GetPrim())
    
    # Parent Xform for the living room
    living_room = UsdGeom.Xform.Define(stage, '/LivingRoom')
    
    # Create Floor
    floor = UsdGeom.Mesh.Define(stage, '/LivingRoom/Floor')
    floor.CreatePointsAttr([
        (-5, 0, -5),
        (5, 0, -5),
        (5, 0, 5),
        (-5, 0, 5)
    ])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr([
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0)
    ])
    floor.CreateExtentAttr().Set(floor.GetPointsAttr().Get())
    
    # Create Walls
    walls = [
        {'name': 'Wall_Back', 'points': [(-5,0,-5), (5,0,-5), (5,3,-5), (-5,3,-5)]},
        {'name': 'Wall_Front', 'points': [(-5,0,5), (5,0,5), (5,3,5), (-5,3,5)]},
        {'name': 'Wall_Left', 'points': [(-5,0,-5), (-5,0,5), (-5,3,5), (-5,3,-5)]},
        {'name': 'Wall_Right', 'points': [(5,0,-5), (5,0,5), (5,3,5), (5,3,-5)]},
    ]
    
    for wall in walls:
        wall_mesh = UsdGeom.Mesh.Define(stage, f"/LivingRoom/{wall['name']}")
        wall_mesh.CreatePointsAttr(wall['points'])
        wall_mesh.CreateFaceVertexCountsAttr([4])
        wall_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        wall_mesh.CreateNormalsAttr([
            (0, 0, 1) if 'Back' in wall['name'] else
            (0, 0, -1) if 'Front' in wall['name'] else
            (1, 0, 0) if 'Left' in wall['name'] else
            (-1, 0, 0)
        ] * 4)
        wall_mesh.CreateExtentAttr().Set(wall_mesh.GetPointsAttr().Get())
    
    # Create a simple Sofa (represented as a box)
    sofa = UsdGeom.Cube.Define(stage, '/LivingRoom/Sofa')
    sofa.AddTranslateOp().Set(Gf.Vec3f(-2, 0, 0))
    sofa.AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
    
    # Create a Coffee Table (another box)
    table = UsdGeom.Cube.Define(stage, '/LivingRoom/CoffeeTable')
    table.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
    table.AddScaleOp().Set(Gf.Vec3f(1, 0.5, 1))
    
    # Create a Lamp (cylinder)
    lamp = UsdGeom.Cylinder.Define(stage, '/LivingRoom/Lamp')
    lamp.AddTranslateOp().Set(Gf.Vec3f(2, 0, 2))
    lamp.AddScaleOp().Set(Gf.Vec3f(0.2, 1.5, 0.2))
    
    # Add a Light Source (Point Light using UsdLux.PointLight)
    from pxr import UsdLux  # 이미 임포트 되어 있는지 확인하세요

    light = UsdLux.PluginLight.Define(stage, '/LivingRoom/Light')
    light.AddTranslateOp().Set(Gf.Vec3f(0, 3, 0))
    light.CreateVisibilityAttr('inherited')  # 수정된 부분
    
    # Optionally, add materials (using UsdPreviewSurface)
    from pxr import UsdShade
    
    # Floor Material
    floor_material = UsdShade.Material.Define(stage, '/LivingRoom/Materials/FloorMaterial')
    floor_shader = UsdShade.Shader.Define(stage, '/LivingRoom/Materials/FloorMaterial/Shader')
    floor_shader.CreateIdAttr('UsdPreviewSurface')
    floor_shader.CreateInput('baseColor', Sdf.ValueTypeNames.Color3f).Set((0.8, 0.7, 0.6))
    
    # 'surface' 출력 가져오기
    surface_output = floor_material.CreateSurfaceOutput()
    shader_surface_output = floor_shader.GetOutput('surface')
    
    # shader_surface_output이 유효한지 확인
    if not shader_surface_output:
        print("Error: shader_surface_output이 유효하지 않습니다.")
    else:
        print(f"shader_surface_output이 유효합니다: {shader_surface_output.GetAttr().GetName()}")
        # 올바르게 연결하기
        surface_output.ConnectToSource(shader_surface_output)
    
    # 머티리얼을 바닥에 바인딩
    floor.GetPrim().GetRelationship("material:binding").AddTarget(floor_material.GetPath())
    
    # Assign materials similarly for walls, sofa, etc. (omitted for brevity)
    
    # Save the stage
    stage.GetRootLayer().Save()
    print(f"Living room scene created and saved to {output_file}")

if __name__ == "__main__":
    import pxr
    print('version', pxr.Usd.GetVersion())

    output_usd_file = "living_room.usd"
    create_living_room_scene(output_usd_file)
