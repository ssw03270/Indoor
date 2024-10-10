import numpy as np
from shapely.geometry import LineString, Point, Polygon

def round_to_90_degrees(vector):
    """Rounds a vector to the nearest 90 degrees."""
    x, y = vector
    angle = np.arctan2(y, x)
    rounded_angle = np.round(angle / (np.pi / 2)) * (np.pi / 2)
    return np.array([np.cos(rounded_angle), np.sin(rounded_angle)])

def round_to_45_degrees(vector):
    """Rounds a vector to the nearest 45 degrees."""
    x, y = vector
    angle = np.arctan2(y, x)
    rounded_angle = np.round(angle / (np.pi / 4)) * (np.pi / 4)
    return np.array([np.cos(rounded_angle), np.sin(rounded_angle)])

def get_direction_vector(poly, floor_exterior, angle, cx, cz, threshold=0.1, ax=None):
    angle_deg = np.degrees(angle)[0] % 360
    is_90_degree = (angle_deg % 90 == 0)
    
    if is_90_degree:
        # Objects rotated at multiples of 90 degrees
        directions = {
            0: np.array([1, 0]),     # 0 degrees (East)
            90: np.array([0, 1]),    # 90 degrees (North)
            180: np.array([-1, 0]),  # 180 degrees (West)
            270: np.array([0, -1])   # 270 degrees (South)
        }
        close_walls = []

        for dir_angle, dir_vector in directions.items():
            # Cast a ray from object center in the direction
            ray = LineString([Point(cx, cz), Point(cx + dir_vector[0] * 10, cz + dir_vector[1] * 10)])
            
            intersection_object = ray.intersection(poly.boundary)
            exterior_point = intersection_object - Point(cx, cz)

            ray = LineString([exterior_point, Point(exterior_point.x + dir_vector[0] / 10, exterior_point.y + dir_vector[1] / 10)])
            intersection_floor = ray.intersection(floor_exterior)

            if intersection_floor.is_empty:
                ray = LineString([exterior_point, Point(exterior_point.x - dir_vector[0] / 10, exterior_point.y - dir_vector[1] / 10)])
                intersection_floor = ray.intersection(floor_exterior)

            if not intersection_floor.is_empty:
                # Calculate distance to wall
                distance = intersection_floor.distance(intersection_object)
                if distance < threshold:
                    close_walls.append(dir_angle)
                    if ax is not None:
                        x, y = intersection_object.xy
                        ax.plot(x, y, 'ro', markersize=5)  # Mark intersection points

        num_close_walls = len(close_walls)
        
        if num_close_walls == 0:
            # Find the closest wall
            closest_wall = None
            min_distance = float('inf')
            for dir_angle, dir_vector in directions.items():
                ray = LineString([Point(cx, cz), Point(cx + dir_vector[0] * 10, cz + dir_vector[1] * 10)])
                intersection = ray.intersection(floor_exterior)
                if not intersection.is_empty:
                    distance = Point(cx, cz).distance(intersection)
                    if distance < min_distance and distance < threshold * 3:
                        min_distance = distance
                        closest_wall = dir_angle

            # Set direction opposite to the closest wall
            if closest_wall is not None:
                opposite_angle = (closest_wall + 180) % 360
                return directions[opposite_angle]
            else:
                # Default direction if no wall is found
                width, height = poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]
                room_center_x = (floor_exterior.bounds[0] + floor_exterior.bounds[2]) / 2
                room_center_z = (floor_exterior.bounds[1] + floor_exterior.bounds[3]) / 2

                if width > height:
                    # Longer axis is along X, so direction is along Y
                    if cz < room_center_z:
                        return directions[90]  # North
                    else:
                        return directions[270]  # South
                else:
                    # Longer axis is along Y, so direction is along X
                    if cx < room_center_x:
                        return directions[0]  # East
                    else:
                        return directions[180]  # West

        elif num_close_walls == 1:
            # If only one wall is close, assign direction opposite to that wall
            opposite_angle = (close_walls[0] + 180) % 360
            direction_vector = directions[opposite_angle]
            return direction_vector

        elif num_close_walls == 2:
            # If two walls are close, assign direction along the longer axis orthogonal to that axis
            width, height = poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]
            if width > height:
                # Longer axis is along X, so direction is along Y
                if 90 not in close_walls:
                    return directions[90]
                else:
                    return directions[270]
            else:
                # Longer axis is along Y, so direction is along X
                if 0 not in close_walls:
                    return directions[0]
                else:
                    return directions[180]

        elif num_close_walls == 3:
            # If three walls are close, assign direction to the remaining direction
            remaining_directions = set(directions.keys()) - set(close_walls)
            if remaining_directions:
                direction_angle = remaining_directions.pop()
                return directions[direction_angle]
            else:
                # Default direction if all directions are close
                return np.array([0, 1])

        elif num_close_walls == 4:
            # If all walls are close, choose the direction closest to the origin
            vectors = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
            distances = [np.linalg.norm((cx, cz) + v * threshold) for v in vectors]
            min_index = np.argmin(distances)
            return vectors[min_index]
        else:
            # If no walls are close, default to pointing towards the origin
            return round_to_90_degrees(np.array([-cx, -cz]))

    else:
        # Objects rotated at multiples of 45 degrees
        return round_to_45_degrees(np.array([-cx, -cz]))
    
def vector_to_angle(dx, dz):
    """Converts a direction vector to an angle in degrees."""
    angle_rad = np.arctan2(dz, dx)
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg

def create_rectangle(x, y, width, height):
    # 사각형의 반 너비와 반 높이를 구합니다.
    half_width = width / 2
    half_height = height / 2
    
    # 사각형의 네 꼭짓점 좌표를 원점 기준으로 계산합니다.
    corners = [
        (x + -half_width, y + -half_height),
        (x + half_width, y + -half_height),
        (x + half_width, y + half_height),
        (x + -half_width, y + half_height)
    ]
    
    # 다각형을 생성합니다.
    rotated_rectangle = Polygon(corners)
    
    return rotated_rectangle

def check_ray_collision(rects, floor_exterior, threshold=[[0.1, 'adjacent'], [0.5, 'proximal'], [np.inf, 'distant']]):
    """
    동일한 레벨의 rect들에 대해 상하좌우로 ray를 쏘고 충돌을 확인합니다.
    
    :param rects: rect 정보를 담은 리스트
    :param threshold: 충돌 판정을 위한 거리 임계값
    :return: 충돌이 발생한 경우 True, 그렇지 않으면 False
    """
    directions = [(0, 1, 'top'), (0, -1, 'bottom'), (1, 0, 'right'), (-1, 0, 'left')]  # 상, 하, 우, 좌
    distance_list = []
    
    for i, rect1 in enumerate(rects):
        distance_types = {}

        for dx, dy, dir in directions:
            distance_type = None
            min_distance = np.inf
            for th, _type in threshold:
                for j, rect2 in enumerate(rects):
                    if i != j and rect1['level'] == rect2['level']:
                        # rect1의 경계 계산
                        left = rect1['x'] - rect1['width'] / 2
                        right = rect1['x'] + rect1['width'] / 2
                        top = rect1['y'] + rect1['height'] / 2
                        bottom = rect1['y'] - rect1['height'] / 2
                        
                        # 방향에 따라 ray_start 설정
                        if dx > 0:
                            start_point = Point(right, rect1['y'])
                            ray = LineString([Point(rect1['x'], rect1['y']), Point(right + dx * th, rect1['y'] + dy * th)])
                        elif dx < 0:
                            start_point = Point(left, rect1['y'])
                            ray = LineString([Point(rect1['x'], rect1['y']), Point(left + dx * th, rect1['y'] + dy * th)])
                        elif dy > 0:
                            start_point = Point(rect1['x'], top)
                            ray = LineString([Point(rect1['x'], rect1['y']), Point(rect1['x'] + dx * th, top + dy * th)])
                        else:  # dy < 0
                            start_point = Point(rect1['x'], bottom)
                            ray = LineString([Point(rect1['x'], rect1['y']), Point(rect1['x'] + dx * th, bottom + dy * th)])

                        intersection = ray.intersection(create_rectangle(rect2['x'], rect2['y'], rect2['width'], rect2['height']))
                        if intersection:
                            distance = start_point.distance(intersection)
                            if distance < min_distance:
                                distance_type = _type
                                min_distance = distance

                        intersection = ray.intersection(floor_exterior)
                        if intersection:
                            distance = start_point.distance(intersection)
                            if distance < min_distance:
                                distance_type = _type
                                min_distance = distance
            min_distance = round(min_distance, 2)

            if distance_type == None:
                distance_type = threshold[-1][-1]
                min_distance = 1

            distance_types[dir] = [distance_type, min_distance]
        distance_list.append(distance_types)

    return distance_list
            