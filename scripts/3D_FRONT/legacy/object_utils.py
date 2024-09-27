import json

class Scene:
    def __init__(self, file_path):
        self.file_path = file_path
        self.json_data = json.load(open(file_path, 'r', encoding='utf-8'))

        self.uid = self.json_data['uid']
        self.jobid = self.json_data['jobid']
        self.design_version = self.json_data['design_version']
        self.code_version = self.json_data['code_version']
        self.north_vector = self.json_data['north_vector']
        self.furniture = [furniture for furniture in self.json_data['furniture'] if 'valid' in furniture and furniture['valid'] == True]
        self.mesh = self.json_data['mesh']
        self.material = self.json_data['material']
        self.lights = self.json_data['lights']
        self.extension = self.json_data['extension']
        self.scene = self.json_data['scene']
        self.groups = self.json_data['groups']
        self.materialList = self.json_data['materialList']
        self.version = self.json_data['version']

        self.rooms = [Room(room) for room in self.scene['room']]

    def __str__(self):
        return f"Scene(file_path={self.file_path})"

class Room:
    def __init__(self, room_data):
        self.room_data = room_data

        self.room_type = self.room_data['type']
        self.instanceid = self.room_data['instanceid']
        self.size = self.room_data['size']
        self.pos = self.room_data['pos']
        self.rot = self.room_data['rot']
        self.scale = self.room_data['scale']
        self.children = [Furniture(child) for child in self.room_data['children']]
        self.empty = self.room_data['empty']


    def __str__(self):
        output = f"Room(room_type={self.room_type}, instanceid={self.instanceid})"
        return output

class Furniture:
    def __init__(self, furniture_data):
        self.furniture_data = furniture_data

        self.ref = self.furniture_data['ref']
        self.instanceid = self.furniture_data['instanceid']
        self.pos = self.furniture_data['pos']
        self.rot = self.furniture_data['rot']
        self.scale = self.furniture_data['scale']
        self.replace_jid = self.furniture_data['replace_jid'] if 'replace_jid' in self.furniture_data else None
        self.replace_bbox = self.furniture_data['replace_bbox'] if 'replace_bbox' in self.furniture_data else None

    def __str__(self):
        output = f"Furniture(ref={self.ref}, instanceid={self.instanceid}, replace_jid={self.replace_jid})"
        return output
