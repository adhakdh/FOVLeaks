import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import re
import json


def cal_file_accuracy_website(Result_dict):
    
    true_key_list = Result_dict.keys()

    
    top1_result_dict = {key: [] for key in true_key_list}
    top3_result_dict = {key: [] for key in true_key_list}
    tmp_len_list = []
    for key, value in Result_dict.items():  
        for v_item in value:
            top1_result_dict[key].append(v_item[1][0])  
            tmp_len = min(len(v_item[1][1]), len(key)) + 2 
            tmp_len2 = min(len(v_item[1][1]), len(key)) + 2 
            tmp_len_list.append(tmp_len)
            tmp_len_list2.append(tmp_len2)
            top3_string = ''.join(tmp_key[-1] for tmp_key in v_item[1:4]) 
            match_string = lcs(key + "1", top3_string)
            accuracy = len(match_string) /len(key)
            top3_result_dict[key].append(accuracy)
                
        len_dict = len(value)
        if len_dict == 0:
            top1_result_dict[key] = 0
            top3_result_dict[key] = 0
        else:
            top3_result_dict[key] = sum(top3_result_dict[key]) / len_dict
            top1_result_dict[key] = sum(top1_result_dict[key]) / len_dict

    return top1_result_dict, top3_result_dict, tmp_len_list


def cal_file_accuracy(Result_dict):
    
    true_key_list = Result_dict.keys()

    
    top1_result_dict = {key: [] for key in true_key_list}
    top3_result_dict = {key: [] for key in true_key_list}
    tmp_len_list = []
    for key, value in Result_dict.items():  
        for v_item in value:
            top1_result_dict[key].append(v_item[1][0])  
            tmp_len = min(len(v_item[1][1]), len(key)) + 2 
            tmp_len2 = min(len(key), len(key)) + 2 
            tmp_len_list.append([tmp_len,tmp_len2])
            top3_string = ''.join(tmp_key[-1] for tmp_key in v_item[1:4]) 
            match_string = lcs(key, top3_string)
            accuracy = len(match_string) /len(key)
            top3_result_dict[key].append(accuracy)
                
        len_dict = len(value)
        if len_dict == 0:
            top1_result_dict[key] = 0
            top3_result_dict[key] = 0
        else:
            top3_result_dict[key] = sum(top3_result_dict[key]) / len_dict
            top1_result_dict[key] = sum(top1_result_dict[key]) / len_dict

    return top1_result_dict, top3_result_dict, tmp_len_list


def lcs(str1, str2):
    m, n = len(str1), len(str2)
    dp = [["" for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                dp[i+1][j+1] = dp[i][j] + str1[i]
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j], key=len)
    
    return dp[m][n]

def cal_accuracy_password(true_key_string, save_string_list):
    accuracy_list = []
    for save_string in save_string_list:
        match_string = lcs(true_key_string, save_string)
        accuracy = len(match_string) /len(true_key_string)
        accuracy_list.append([accuracy,save_string] )

    accuracy_list.sort(key=lambda x: x[0], reverse=True)
    return accuracy_list

def cal_accuracy_website(true_key_string, save_string_list):
    replacement_map = {
        '.com': '1', 
    }
    for key,value in replacement_map.items():
        true_key_string = true_key_string.replace(key, value)


    accuracy_list = []
    for save_string in save_string_list:
        for key,value in replacement_map.items():
            save_string = save_string.replace(key, value)
        match_string = lcs(true_key_string, save_string)
        accuracy = len(match_string) /len(true_key_string)
        accuracy_list.append([accuracy,save_string] )

    accuracy_list.sort(key=lambda x: x[0], reverse=True)
    return accuracy_list



def cal_accuracy_31key(save_string_list):
    true_key_list = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                     'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 
                     '1', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', 
                     '2', '3']
    replacement_map = {
        'shift': '1', 
        'Shift': '1', 
        'Space': '2', 
        'space': '2', 
        'enter': '3',
        'Enter': '3',
        'S9': '.',
        's9': '.',
    }

    accuracy_list = []
    for save_string in save_string_list:
        for key,value in replacement_map.items():
            save_string = save_string.replace(key, value)
        match_string = lcs(true_key_list, save_string)
        accuracy = len(match_string) /len(true_key_list)
        accuracy_list.append([accuracy,save_string] )

    accuracy_list.sort(key=lambda x: x[0], reverse=True)
    return accuracy_list

def get_bast_scale(ALL_intersection_point_list, keyboard_on_xy):
    scales = np.arange(0.4, 0.6, 0.01).tolist()
    bestScale = 0
    maxKeys = 0
    minDist = float('inf')
    for scale in scales:
        anchor_x = ALL_intersection_point_list[-1][0]  - keyboard_on_xy[20][0]*scale  
        anchor_y = ALL_intersection_point_list[-1][1]
        anchor_position = [anchor_x, anchor_y, 0]
        keyboard_project_viewMatrix = get_anchor_viewMatrix(anchor_position, 0, scale)
        keyboard_on_xy_1 = np.array([global_to_local_point(keypoint[:3], keyboard_project_viewMatrix) for keypoint in keyboard_on_xy])

        
        key_dx = 0.5 * abs(keyboard_on_xy_1[0][0] - keyboard_on_xy_1[1][0])
        key_dy = 0.5 * abs(keyboard_on_xy_1[0][1] - keyboard_on_xy_1[11][1])
        
        
        key_name = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'delete', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Enter', 'shift', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', ' ', '!123', '.com', 'Space', ' ', ' ', ' ']
        
        keyboard_xy_2d = keyboard_on_xy_1[:, :2]
        input_points = np.array(ALL_intersection_point_list)[:, :2]
        best_results = optimize_position_by_hit_count_match_and_dist2(keyboard_xy_2d, input_points, key_name, key_dx, key_dy, search_range=0.02, step=0.001)  

        total_dists = [result[1] for result in best_results]
        min_index = total_dists.index(min(total_dists))
        hit_count, total_dist, matched_keys, [dx, dy] = best_results[min_index]

        if hit_count >= maxKeys and total_dist < minDist:
            maxKeys = hit_count
            minDist = total_dist
            bestScale = scale

    return bestScale


def motify_if_website(input_string):
    replacement_map = {
        'S9': 'enter', 
        's9': 'enter',
        'delete': 'p',
        '/': 'm', 
        'web': 'm', 
        'shift': 'z',
        'Space': '.com',
        '!123': '.com',
    }
    for key,value in replacement_map.items():
        input_string = input_string.replace(key, value)
    save_string = input_string
    return save_string

def motify_if_word(input_string):
    
    replacement_map = {
        'delete': 'p',
        'enter': 'l',
        '/': 'l',
        '.': 'l',  
        'web': 'm', 
        'shift': 'z', 
        '.com': 'z',
        '!123': 'z',
        'space': '?',
    }
    for key,value in replacement_map.items():
        input_string = input_string.replace(key, value)
    save_string = input_string
    return save_string


def motify_if_31key(input_string):
    replacement_map = {
        'S9': 'enter',
        's9': 'enter', 
        'web': 'space', 
        'set': 'space', 
        'cancel': '.',
        '.com': 'z',
        '!123': 'shift',
    }
    for key,value in replacement_map.items():
        input_string = input_string.replace(key, value)
    save_string = input_string
    return save_string

def motify_if_PIN(input_string):
    replacement_map = {
        'q': '1', 'w': '2', 'e': '3', 'r': '4',
        't': '5', 'y': '6', 'u': '7', 'i': '8',
        'o': '9', 'p': '0',
        'a': '1', 's': '2', 'd': '3', 'f': '4',
        'g': '5', 'h': '6', 'j': '7', 'k': '8',
        'l': '9'
    }
    num_string = ""

    special_strings = ["enter", "shift", "delete", '!123', '.com', 'space']
    for special in special_strings:
        input_string = input_string.replace(special, "?")

    for leter_string in input_string:
        if leter_string in replacement_map:
            leter_string = replacement_map[leter_string]
        num_string += leter_string
    save_string = num_string
    return save_string


def motify_if_password(input_string):
    replacement_map = {
        'delete': 'p',
        'enter': 'l',
        '/': 'l',
        '.': 'l',
        '.com': '!123',
        'shift': '!123',
        'web': 'l', 
        'set': 'l', 
        'cancel': 'l',
    }
    for key,value in replacement_map.items():
        input_string = input_string.replace(key, value)


    if "!123" not in input_string:
        return input_string
    replacement_map = {
        'q': '1', 'w': '2', 'e': '3', 'r': '4',
        't': '5', 'y': '6', 'u': '7', 'i': '8',
        'o': '9', 'p': '0',
        'a': '1', 's': '2', 'd': '3', 'f': '4',
        'g': '5', 'h': '6', 'j': '7', 'k': '8',
        'l': '9'
    }

    save_string_list = input_string.split("!123")
    if len(save_string_list) > 1 and save_string_list[1]:
        num_string = ""
        for leter_string in save_string_list[1]:
            if leter_string in replacement_map:
                leter_string = replacement_map[leter_string]
            num_string += leter_string
        save_string_list[1] = num_string 
    save_string = "".join(save_string_list)  
    return save_string
   
def get_cross_point(light_position1, light_position2, project_anchor, project_normal_vector):
    
    d = light_position2 - light_position1
    n_dot_p0 = np.dot(project_normal_vector, project_anchor)    
    t = (n_dot_p0 - np.dot(project_normal_vector, light_position1)) / np.dot(project_normal_vector, d)
    intersection_point = light_position1 + t * d
    
    return intersection_point


def optimize_position_by_hit_count_match_and_dist2(keyboard_xy, input_points, key_name, key_dx, key_dy, search_range, step):
    
    letters = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '.com','!123']
    key_rects = []
    letter_key_xy = []
    letter_key_xy_dic = {}
    letters_list = []
    for key in key_name:
        if key in letters:
            dx = key_dx
            dy = key_dy
            key_rects.append((dx, dy))
            letter_key_xy.append(keyboard_xy[key_name.index(key), :2])  
            letter_key_xy_dic[key] = keyboard_xy[key_name.index(key), :2]  
            letters_list.append(key)  
    offsets_x = np.arange(-search_range, search_range + step, step)
    offsets_y = np.arange(-search_range, search_range + step, step)
    
    best_offsets = []
    letter_key_xy = np.array(letter_key_xy)  
    for dx in offsets_x:
        for dy in offsets_y:
            offset_keyboard = letter_key_xy + np.array([dx, dy])
            hit_count = 0
            for point in input_points[:-1]:  
                for i, center in enumerate(offset_keyboard):
                    width, height = key_rects[i]
                    if abs(point[0] - center[0]) < width and abs(point[1] - center[1]) < height:
                        hit_count += 1



            
            manhattan_distances = np.abs(input_points[:, None, :] - offset_keyboard[None, :, :])
            relative_distances = manhattan_distances.copy()
            relative_distances[:, :, 0] /= key_dx  
            relative_distances[:, :, 1] /= key_dy  
            relative_manhattan_distances = np.sum(relative_distances, axis=-1)
            min_distances = np.min(relative_manhattan_distances, axis=1)
            total_dist = np.sum(min_distances)

            
            

            
            matched_keys = match_points_to_keys(input_points[:-1], offset_keyboard, letters_list, key_dx, key_dy)  

            
            best_offsets.append((hit_count, total_dist, matched_keys, [dx, dy]))

    best_results = classify_by_count_dist(best_offsets)
    return best_results 



def optimize_position_by_hit_count_match_and_dist(keyboard_xy, input_points, key_name, key_dx, key_dy, search_range, step):
    
    letters = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '.com','!123']
    key_rects = []
    letter_key_xy = []
    letter_key_xy_dic = {}
    letters_list = []
    for key in key_name:
        if key in letters:
            dx = key_dx
            dy = key_dy
            key_rects.append((dx, dy))
            letter_key_xy.append(keyboard_xy[key_name.index(key), :2])  
            letter_key_xy_dic[key] = keyboard_xy[key_name.index(key), :2]  
            letters_list.append(key)  
    offsets_x = np.arange(-search_range, search_range + step, step)
    offsets_y = np.arange(-search_range, search_range + step, step)
    
    best_offsets = []
    letter_key_xy = np.array(letter_key_xy)  
    for dx in offsets_x:
        for dy in offsets_y:
            offset_keyboard = letter_key_xy + np.array([dx, dy])
            hit_count = 0
            for point in input_points[:-1]:  
                for i, center in enumerate(offset_keyboard):
                    width, height = key_rects[i]
                    if abs(point[0] - center[0]) < width and abs(point[1] - center[1]) < height:
                        hit_count += 1

            
            total_dist = np.sum(np.min(np.abs(input_points[:, None, :] - offset_keyboard[None, :, :]), axis=1))

            
            matched_keys = match_points_to_keys(input_points, keyboard_xy + np.array([dx, dy]), key_name, key_dx, key_dy)
            
            
            best_offsets.append((hit_count, total_dist, matched_keys, [dx, dy]))

    best_results = classify_by_count_dist(best_offsets)
    return best_results 

def classify_by_count_dist(best_offsets):
    
    max_hit_count = max(offset[0] for offset in best_offsets)
    
    
    max_count_results = [result for result in best_offsets if result[0] == max_hit_count]

    
    grouped_results = {}
    for hit_count, total_dist, matched_keys, offset in max_count_results:
        matched_keys_tuple = tuple(matched_keys)  
        if matched_keys_tuple not in grouped_results:
            grouped_results[matched_keys_tuple] = []
        grouped_results[matched_keys_tuple].append((total_dist, offset))

    
    best_results = []
    for keys, results in grouped_results.items():
        best_result = min(results, key=lambda x: x[0])  
        best_results.append((max_hit_count, best_result[0], keys, best_result[1]))

    return best_results  



def optimize_keyboard_position(keyboard_xy, input_points, initial_anchor, search_range=0.02, step=0.001):
    """
    用曼哈顿距离优化 keyboard_xy 相对于 input_points 的偏移位置。
    """
    min_total_dist = float('inf')
    best_offset = initial_anchor[:2]

    offsets_x = np.arange(-search_range, search_range + step, step)
    offsets_y = np.arange(-search_range, search_range + step, step)

    for dx in offsets_x:
        for dy in offsets_y:
            offset_keyboard = keyboard_xy + np.array([dx, dy])

            
            manhattan_dists = np.abs(input_points[:, None, 0] - offset_keyboard[None, :, 0]) + \
                            np.abs(input_points[:, None, 1] - offset_keyboard[None, :, 1])
            min_dists = np.min(manhattan_dists, axis=1)
            total_dist = np.sum(min_dists)

            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_offset = [dx, dy]

    return [best_offset[0], best_offset[1], 0]


def optimize_position_by_hit_count(keyboard_xy, input_points, key_name, key_dx, key_dy,
                                    initial_anchor, search_range=0.005, step=0.001):
    
    
    key_rects = []
    for key in key_name:
        if key.lower() == 'space':
            dx = key_dx * 6
        elif key.lower() in ['enter']:
            dx = key_dx * 2
        else:
            dx = key_dx
        dy = key_dy
        key_rects.append((dx, dy))

    offsets_x = np.arange(-search_range, search_range + step, step)
    offsets_y = np.arange(-search_range, search_range + step, step)
    
    
    best_offset = initial_anchor[:2]
    max_hit_count = -1

    for dx in offsets_x:
        for dy in offsets_y:
            offset_keyboard = keyboard_xy + np.array([dx, dy])

            hit_count = 0
            for point in input_points:
                for i, center in enumerate(offset_keyboard):
                    width, height = key_rects[i]
                    if abs(point[0] - center[0]) < width and abs(point[1] - center[1]) < height:
                        hit_count += 1
                        break  
            if hit_count >= max_hit_count:
                max_hit_count = hit_count
                best_offset = [dx, dy]
                

    return [best_offset[0], best_offset[1], 0]

def get_points_in_keys(input_points, keyboard_xy, key_name, key_dx, key_dy):
    key_rects = []
    for key in key_name:
        if key.lower() == 'space':
            dx = key_dx * 5
        elif key.lower() in ['enter', '!123']:    
            dx = key_dx * 2
        else:
            dx = key_dx
        dy = key_dy
        key_rects.append((dx, dy))

    results = []
    for idx, point in enumerate(input_points):
        for i, center in enumerate(keyboard_xy):
            width, height = key_rects[i]
            if abs(point[0] - center[0]) < width and abs(point[1] - center[1]) < height:
                results.append((idx + 1, key_name[i]))  
                break  
    return results  




def match_points_to_keys(input_points, keyboard_xy, key_name, key_dx, key_dy):
    key_rects = []
    for key in key_name:
        if key.lower() == 'space':
            dx = key_dx * 5
        elif key.lower() in ['enter', '!123']:    
            dx = key_dx * 2
        else:
            dx = key_dx
        dy = key_dy
        key_rects.append((dx, dy))

    results = []
    for idx, point in enumerate(input_points):
        matched_key = "None"
        closest_key = None
        min_distance = float('inf')

        
        for i, center in enumerate(keyboard_xy):
            distance = abs(point[0] - center[0]) + abs(point[1] - center[1])
            if distance < min_distance:
                min_distance = distance
                closest_key = i

        matched_key = key_name[closest_key]
        results.append((idx + 1, matched_key))  

    return results  


def load_json_data(input_path='stare_when_input.json'):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_view_matrix_from_head_pose(position_xyz,quat_xyzw):
    x, y, z, w = quat_xyzw
    px, py, pz = position_xyz

    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])

    
    R_T = R.T  
    t = np.array([px, py, pz])
    translation = -np.dot(R_T, t)

    
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R_T
    view_matrix[:3, 3] = translation

    return view_matrix




def merge_close_points(json_path, output_path, time_log_path):
    time_threshold=0.3
    with open(json_path, 'r') as f:
        data = json.load(f)

    times = data["IMUTime"]
    z_dirs = data["Head_Z_direction"]
    positions = data["Head_position"]
    orientations = data["Head_orientation"]

    merged_data = {
        "IMUTime": [],
        "Head_Z_direction": [],
        "Head_position": [],
        "Head_orientation": []
    }

    if not times:
        print("No data found.")
        return

    current_group = [0]
    time_ranges = []

    for i in range(1, len(times)):
        if times[i] - times[i - 1] < time_threshold:
            current_group.append(i)
        else:
            merged_data["IMUTime"].append(np.mean([times[j] for j in current_group]))
            merged_data["Head_Z_direction"].append(np.mean([z_dirs[j] for j in current_group], axis=0).tolist())
            merged_data["Head_position"].append(np.mean([positions[j] for j in current_group], axis=0).tolist())
            merged_data["Head_orientation"].append(np.mean([orientations[j] for j in current_group], axis=0).tolist())
            time_ranges.append((times[current_group[0]], times[current_group[-1]]))
            current_group = [i]

    
    if current_group:
        merged_data["IMUTime"].append(np.mean([times[j] for j in current_group]))
        merged_data["Head_Z_direction"].append(np.mean([z_dirs[j] for j in current_group], axis=0).tolist())
        merged_data["Head_position"].append(np.mean([positions[j] for j in current_group], axis=0).tolist())
        merged_data["Head_orientation"].append(np.mean([orientations[j] for j in current_group], axis=0).tolist())
        time_ranges.append((times[current_group[0]], times[current_group[-1]]))

    
    with open(output_path, 'w') as f_out:
        json.dump(merged_data, f_out)
    

    
    time_log = {idx: round(end - start, 3) for idx, (start, end) in enumerate(time_ranges, 1)}
    with open(time_log_path, 'w') as log_file:
        json.dump(time_log, log_file, indent=4)



def read_head_data(filepath):
    head_data_dic = {
        "IMUTime":[],
        "Head_Z_direction": [],
        "Head_position": [],
        "Head_orientation": []
    }
    with open(filepath, 'r') as f:
        for line in f:
            if "IMUTime" in line:
                
                

                line_item = line.split('IMUTime:')[-1].split(',Vel:')[0]
                head_data_dic["IMUTime"].append(float(line_item))

                line_item = line.split('Head_Z_direction:(')[-1].split(')')[0]
                head_data_dic["Head_Z_direction"].append([float(item) for item in line_item.split(',')])

                line_item = line.split('Head_position:(')[-1].split(')')[0]
                head_data_dic["Head_position"].append([float(item) for item in line_item.split(',')])

                line_item = line.split('Head_orientation:(')[-1].split(')')[0]
                head_data_dic["Head_orientation"].append([float(item) for item in line_item.split(',')])

    return head_data_dic


def compute_derivative(z_dirs, times, max_clip=2.0):
    dt = np.array([(t2 - t1) for t1, t2 in zip(times[:-1], times[1:])])
    dz = np.diff(z_dirs, axis=0)
    deriv = np.linalg.norm(dz, axis=1) / dt
    deriv = np.clip(deriv, 0, max_clip)
    return deriv, dt

def detect_events(z_dirs, derivative, times):
    stare_thresh=0.04 
    tilt_rate_thresh=0.1  
    min_tilt_delta=0.1   
    events = {'look_up': [], 'look_down': [], 'stare': []}
    
    
    y_vals = z_dirs[:, 1]
    dt = np.array([(t2 - t1) for t1, t2 in zip(times[:-1], times[1:])])
    dy = np.diff(y_vals)
    dy_dt = dy / dt

    
    i = 0
    while i < len(dy_dt):
        if dy_dt[i] > tilt_rate_thresh:
            start = i
            while i < len(dy_dt) and dy_dt[i] > tilt_rate_thresh:
                i += 1
            end = i
            delta_y = y_vals[end] - y_vals[start]
            if delta_y > min_tilt_delta:
                events['look_down'].append((start, end))
        elif dy_dt[i] < -tilt_rate_thresh:
            start = i
            while i < len(dy_dt) and dy_dt[i] < -tilt_rate_thresh:
                i += 1
            end = i
            delta_y = y_vals[end] - y_vals[start]
            if delta_y < -min_tilt_delta:
                events['look_up'].append((start, end))
        else:
            i += 1

    
    stare_start = None
    for i in range(1, len(derivative)):
        if derivative[i] < stare_thresh:
            if stare_start is None:
                stare_start = i
        else:
            if stare_start is not None and i - stare_start > 3:    
                events['stare'].append((stare_start, i))
            stare_start = None
    return events, dy_dt



def save_events(head_data_dic, events, output_path='../lib/savedata/events.json'):
    
    selected_data_dic = {
        "look_down": [],
        "look_up": [],
        "stare": [],
    }
    
    for start, end in events['look_down']:
        selected_data_dic["look_down"].append(head_data_dic["IMUTime"][start])
    for start, end in events['look_up']:
        selected_data_dic["look_up"].append(head_data_dic["IMUTime"][start])
    for start, end in events['stare']:
        selected_data_dic["stare"].append((head_data_dic["IMUTime"][start],head_data_dic["IMUTime"][end]))

    with open(output_path, 'w') as json_file:
        json.dump(selected_data_dic, json_file)


def save_stare_when_call_keyboard(head_data_dic, events, output_path='../lib/savedata/stare_when_call_keyboard.json'):
    if not events['look_up'] or not events['look_down']:
        raise ValueError("Insufficient events detected for look_up or look_down.")
        return

    
    look_up_sorted = sorted(events['look_up'], key=lambda x: x[0])
    look_down_sorted = sorted(events['look_down'], key=lambda x: x[0])

    selected_data_dic = {
        "IMUTime": [],
        "Head_Z_direction": [],
        "Head_position": [],
        "Head_orientation": []
    }

    
    i, j = 0, 0
    while i < len(look_up_sorted) and j < len(look_down_sorted):
        up_start, up_end = look_up_sorted[i]
        down_start, down_end = look_down_sorted[j]

        if down_start > up_end:
            
            selected_data_dic["IMUTime"].extend(head_data_dic["IMUTime"][up_end:down_start])
            selected_data_dic["Head_Z_direction"].extend(head_data_dic["Head_Z_direction"][up_end:down_start])
            selected_data_dic["Head_position"].extend(head_data_dic["Head_position"][up_end:down_start])
            selected_data_dic["Head_orientation"].extend(head_data_dic["Head_orientation"][up_end:down_start])
            i += 1
            j += 1
        elif down_start <= up_end:
            j += 1  

    if selected_data_dic["IMUTime"]:
        with open(output_path, 'w') as json_file:
            json.dump(selected_data_dic, json_file)
    else:
        print("No valid segments found between look_up and look_down.")
        exit()


def save_stare_when_input(head_data_dic, events, output_path='../lib/savedata/stare_when_input.json'):
    if not events['look_up'] or not events['look_down'] or not events['stare']:
        raise ValueError("Insufficient events detected for look_up, look_down, or stare.")
        return

    
    
    
    

    
    last_lookdown_end = max(end for _, end in events['look_down'])

    
    selected_data_dic = {
        "IMUTime": [],
        "Head_Z_direction": [],
        "Head_position": [],
        "Head_orientation": []
    }

    
    for start, end in events['stare']:
        if start > last_lookdown_end:
            selected_data_dic["IMUTime"].extend(head_data_dic["IMUTime"][start:end])
            selected_data_dic["Head_Z_direction"].extend(head_data_dic["Head_Z_direction"][start:end])
            selected_data_dic["Head_position"].extend(head_data_dic["Head_position"][start:end])
            selected_data_dic["Head_orientation"].extend(head_data_dic["Head_orientation"][start:end])

    if selected_data_dic["IMUTime"]:  
        with open(output_path, 'w') as json_file:
            json.dump(selected_data_dic, json_file)


def axis_angle_to_homogeneous_rotation_matrix(axis, angle):
    axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    rotvec = angle * np.array(axis_map[axis])
    rotation_matrix_3x3 = R.from_rotvec(rotvec).as_matrix()
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix_3x3
    return rotation_matrix_4x4


def cal_transform(anchor_point, anchor_rotation, scale):
    translation = np.array([
        [1, 0, 0, anchor_point[0]],
        [0, 1, 0, anchor_point[1]],
        [0, 0, 1, anchor_point[2]],
        [0, 0, 0, 1]
    ])
    qx, qy, qz, qw = anchor_rotation
    rotation = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy), 0],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx), 0],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2), 0],
        [0, 0, 0, 1]
    ])
    scaling = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]
    ])

    z_180_rotation = axis_angle_to_homogeneous_rotation_matrix('z', np.pi)  
    y_180_rotation = axis_angle_to_homogeneous_rotation_matrix('y', np.pi)  
    x_90_rotation = axis_angle_to_homogeneous_rotation_matrix('x', np.pi*0.5)  

    
    
    pos_matrix = np.dot(translation, rotation)
    transform_ = np.dot(pos_matrix, scaling)

    return transform_


def axis_angle_to_quaternion(axis, angle):
    axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    rotvec = angle * np.array(axis_map[axis])
    quaternion = R.from_rotvec(rotvec).as_quat()
    return quaternion


def keyboard_z_rotation(keyboard_z_rotation):
    anchor_rotation = axis_angle_to_quaternion("z", np.radians(keyboard_z_rotation)) 
    transform_ = cal_transform([0,0,0], anchor_rotation, 1)
    return transform_

def get_anchor_viewMatrix(keyboard_anchor_position, keyboard_x_rotation, scale):
    anchor_rotation = axis_angle_to_quaternion("x", np.radians(keyboard_x_rotation)) 
    
    
    transform_ = cal_transform(keyboard_anchor_position, anchor_rotation, scale)
    return transform_

def get_vision_light_data(file_path):
    vision_light = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if "1_cameraPosition" in line:
            tmp_dict = {}
            line = line.split(",1_cameraPosition:")[-1]
            tmp_dict['Camera_position'] = np.array([float(coord) for coord in line.split("(")[1].split(")")[0].split(",")])
            tmp_dict['Camera_rotation'] = np.array([float(angle) for angle in line.split(", cameraRotation:(")[-1].split(")")[0].split(",")])
            tmp_dict['Z_direction'] = np.array([float(direction) for direction in line.split(", zDirection:(")[-1].split(")")[0].split(",")])
            viewMatrix = line.split(", viewMatrix:(")[-1].split(")")[0].split(",")
            tmp_dict['viewMatrix'] = np.array([viewMatrix[i:i+4] for i in range(0, len(viewMatrix), 4)], dtype=float)
            vision_light.append(tmp_dict)
    return vision_light


def get_vision_light(input_dir):
    file_list = os.listdir(input_dir)
    vision_light_list = []
    for filename in file_list:
        file_path = os.path.join(input_dir, filename)
        vision_light = get_vision_light_data(file_path)
        vision_light_list.extend(vision_light)
    return vision_light_list



def get_vision_center(line_list_list):
    
    closest_point_list = []
    for line_list_l in line_list_list[0]:
        for line_list_r in line_list_list[1]:
            P1 = line_list_l[0]
            D1 = line_list_l[1]
            P2 = line_list_r[0]
            D2 = line_list_r[1]
            closest_point_on_L1, closest_point_on_L2 = closest_points_on_lines(P1, D1, P2, D2)
            closest_point = np.mean([closest_point_on_L1, closest_point_on_L2], axis=0)
            closest_point_list.append(closest_point)

    
    points_global = np.mean(closest_point_list, axis=0)
    return points_global



def angle_with_z_axis(input_vector):
    
    vector = input_vector.copy()
    z_axis = np.array([0, 0, 1])  
    cosine_angle = np.dot(vector, z_axis) / (np.linalg.norm(vector) * np.linalg.norm(z_axis))
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def local_to_global_vector(normal_vector_local, view_matrix):
    normal_homogeneous = np.append(normal_vector_local, 0)  
    inverse_view_matrix = np.linalg.inv(view_matrix)
    restored_normal = inverse_view_matrix @ normal_homogeneous
    return restored_normal[:3]  

def global_to_local_vector(normal_vector, view_matrix):
    normal_homogeneous = np.append(normal_vector, 0)  
    transformed_normal = view_matrix @ normal_homogeneous
    return transformed_normal[:3]  


def global_to_local_point(line_points, view_matrix):
    point_homogeneous = np.append(line_points, 1)
    transformed_points_view = view_matrix @ point_homogeneous.T
    normalized_coords = transformed_points_view / transformed_points_view[-1]
    return normalized_coords[:3]  

def local_to_global_point(local_point, view_matrix):
    point_homogeneous = np.append(local_point, 1)  
    inverse_view_matrix = np.linalg.inv(view_matrix)
    restored_point_homogeneous = inverse_view_matrix @ point_homogeneous
    normalized_coords = restored_point_homogeneous / restored_point_homogeneous[-1]
    return normalized_coords[:3]


def rotate_points_around_axie(pos_list, angle_degrees, axis='x'):
    
    angle_radians = np.radians(angle_degrees)
    
    
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    
    pos_list = np.array(pos_list)
    rotated_positions = np.dot(pos_list, rotation_matrix.T)
    return rotated_positions

def calculate_view_plane(line_points, view_matrix, projection_matrix):
    point_homogeneous = np.append(line_points, 1)
    transformed_points_view =  view_matrix @ point_homogeneous.T
    transformed_points_proj = projection_matrix @ transformed_points_view
    transformed_points_proj = transformed_points_proj.T
    normalized_coords = transformed_points_proj / transformed_points_proj[-1]

    return normalized_coords[:3]  

def get_points_filename(root_dir, points_list):
    names = {}
    filename_list = os.listdir(root_dir)
    for point in points_list:
        names[point] = []
        for filename in filename_list:
            if filename.startswith(point) and filename.endswith('.txt'):
                names[point].append(os.path.join(root_dir, filename))
    return names


def read_camera_data(file_path):
    camera_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    k = -1
    for line in lines:
        if "1_cameraPosition" in line:
            k += 1
            tmp_dict = {}
            line = line.split(",1_cameraPosition:")[-1]
            tmp_dict['Camera_position'] = np.array([float(coord) for coord in line.split("(")[1].split(")")[0].split(",")])
            
            
            tmp_dict['Camera_rotation'] = np.array([float(angle) for angle in line.split(", cameraRotation:(")[-1].split(")")[0].split(",")])
            tmp_dict['Z_direction'] = np.array([float(direction) for direction in line.split(", zDirection:(")[-1].split(")")[0].split(",")])
            viewMatrix = line.split(", viewMatrix:(")[-1].split(")")[0].split(",")
            tmp_dict['viewMatrix'] = np.array([viewMatrix[i:i+4] for i in range(0, len(viewMatrix), 4)], dtype=float)
            projectionMatrix = line.split(", projectionMatrix:(")[-1].split(")")[0].split(",")
            tmp_dict['projectionMatrix'] = np.array([projectionMatrix[i:i+4] for i in range(0, len(projectionMatrix), 4)], dtype=float)
            camera_data[k]  = tmp_dict
    return camera_data




def parse_camera_data(file_path):
    camera_data = {}
    camera_data_0 = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        k = -1
        for line in lines:
            if "1_cameraPosition" in line:
                k += 1
                tmp_dict = {}
                
                line = line.split(",1_cameraPosition:")[-1].split(",2_cameraPosition:")[0]
                tmp_dict['Camera_position'] = np.array([float(coord) for coord in line.split("(")[1].split(")")[0].split(",")])
                
                
                tmp_dict['Camera_rotation'] = np.array([float(angle) for angle in line.split(", cameraRotation:(")[-1].split(")")[0].split(",")])
                tmp_dict['Z_direction'] = np.array([float(direction) for direction in line.split(", zDirection:(")[-1].split(")")[0].split(",")])
                viewMatrix = line.split(", viewMatrix:(")[-1].split(")")[0].split(",")
                tmp_dict['viewMatrix'] = np.array([viewMatrix[i:i+4] for i in range(0, len(viewMatrix), 4)], dtype=float)
                projectionMatrix = line.split(", projectionMatrix:(")[-1].split(")")[0].split(",")
                tmp_dict['projectionMatrix'] = np.array([projectionMatrix[i:i+4] for i in range(0, len(projectionMatrix), 4)], dtype=float)
                if k == 0:
                    camera_data_0 = tmp_dict 
                else:
                    camera_data[k]  = tmp_dict
    return camera_data, camera_data_0

def calculate_line_equation(point1, point2):
    """计算直线的斜率和截距"""
    if point2[0] - point1[0] == 0:  
        return None, None  
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - m * point1[0]
    return m, b

def calculate_intersection(m1, b1, m2, b2):
    """计算两条直线的交点"""
    if m1 is None or m2 is None:
        return None  
    if m1 == m2:
        return None  
    
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

def find_intersection(unified_points1, unified_points2):
    """计算 measurements1 和 measurements2 的交点"""
    point1_a = unified_points1[0, :2]
    point1_b = unified_points1[-1, :2]
    point2_a = unified_points2[0, :2]
    point2_b = unified_points2[-1, :2]

    
    m1, b1 = calculate_line_equation(point1_a, point1_b)
    m2, b2 = calculate_line_equation(point2_a, point2_b)

    
    intersection = calculate_intersection(m1, b1, m2, b2)
    return intersection

def normalize(v):
    return v / np.linalg.norm(v)


def closest_points_on_lines(P1, D1, P2, D2):
    
    P1 = np.array(P1)
    D1 = np.array(D1) / np.linalg.norm(D1)  
    P2 = np.array(P2)
    D2 = np.array(D2) / np.linalg.norm(D2)  

    
    D1_cross_D2 = np.cross(D1, D2)
    denom = np.dot(D1_cross_D2, D1_cross_D2)

    
    if denom == 0:
        print("Lines are parallel.")
        return None  

    
    P2_minus_P1 = P2 - P1
    t = np.dot(np.cross(P2_minus_P1, D2), D1_cross_D2) / denom
    s = np.dot(np.cross(P2_minus_P1, D1), D1_cross_D2) / denom

    
    closest_point_on_L1 = P1 + t * D1
    closest_point_on_L2 = P2 + s * D2

    return closest_point_on_L1, closest_point_on_L2

def cal_cross_angle(viewMatrix):
    
    camera_plane_normal = viewMatrix[0, :3]
    camera_plane_normal = camera_plane_normal / np.linalg.norm(camera_plane_normal)

    zy_plane_normal = np.array([1.0, 0.0, 0.0])

    cosine_angle = np.dot(camera_plane_normal, zy_plane_normal) / (np.linalg.norm(camera_plane_normal) * np.linalg.norm(zy_plane_normal))
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)


    return angle_radians, angle_degrees

def rotate_vector_by_angle(points_3d, angle_degree, y_axis):
    angle_radians = np.radians(angle_degree)  
    
    rotation_vector = angle_radians * y_axis  
    rotation = R.from_rotvec(rotation_vector)
    rotated_point = rotation.apply(points_3d)
    return rotated_point

def cal_points_cross_angle(points, camera_position, camera_x_direction):
    print(points, camera_position)
    vector = np.array([points[0] - camera_position[0], 0, points[2] - camera_position[2]])

    vector_normalized = vector / np.linalg.norm(vector)
    camera_x_direction_normalized = camera_x_direction / np.linalg.norm(camera_x_direction)
    cosine_angle = np.dot(vector_normalized, camera_x_direction_normalized)

    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  
    angle_degrees = np.degrees(angle_radians)
    return angle_radians, angle_degrees

def cal_points_cross_angle_c(points, camera_x_direction):
    vector = points
    camera_x_direction = np.array([1, 0, 0])  

    vector_normalized = vector / np.linalg.norm(vector)
    camera_x_direction_normalized = camera_x_direction / np.linalg.norm(camera_x_direction)
    cosine_angle = np.dot(vector_normalized, camera_x_direction_normalized)

    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  
    angle_degrees = np.degrees(angle_radians)
    return angle_radians, angle_degrees


def transform_to_world_coordinates(point, view_matrix, projection_matrix):   
    point_homogeneous = np.append(point, 1)
    camera_coords_homogeneous = np.dot( np.linalg.inv(projection_matrix), point_homogeneous)
    world_coords = np.dot(np.linalg.inv(view_matrix), camera_coords_homogeneous)
    world_coords = world_coords[:3] / world_coords[3]  
    return world_coords

def read_keyboard(filename):
    keys_list = []
    pos_list = []

    
    with open(filename, 'r') as f:
        for line in f:
            if match := re.search(r'Pos: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', line):
                x_val, y_val, z_val = map(float, match.groups())
                if z_val == 0:  
                    key_name = line.split("Key: key_")[-1].split(", Pos")[0]
                    keys_list.append(key_name)
                    pos_list.append([x_val, y_val, z_val, 1])
    return keys_list, pos_list
