import numpy as np
from lib import global_to_local_point, get_anchor_viewMatrix, rotate_points_around_axie, load_json_data, get_cross_point, get_view_matrix_from_head_pose,optimize_position_by_hit_count_match_and_dist, motify_if_password, motify_if_PIN, motify_if_31key, cal_accuracy_31key, motify_if_website, cal_accuracy_password, cal_accuracy_website, motify_if_word, plot_figure
import numpy as np
import json

def main(head_data_dic, anchor_viewMatrix, project_anchor, project_normal_vector, wether_plot, key_type, scale, TRUR_string):
    ALL_intersection_point_list = []
    for i in range(len(head_data_dic["Head_position"])):
        light_position1 = np.array(head_data_dic["Head_position"][i])
        light_normal_vector = np.array(head_data_dic["Head_Z_direction"][i])
        light_position2 = light_position1 + light_normal_vector * 1  
        light_position1_in_anchor_sys = global_to_local_point(light_position1, anchor_viewMatrix)
        light_position2_in_anchor_sys = global_to_local_point(light_position2, anchor_viewMatrix)
        intersection_point = get_cross_point(light_position1_in_anchor_sys[:3], light_position2_in_anchor_sys[:3], project_anchor, project_normal_vector)
        ALL_intersection_point_list.append(intersection_point)


    if "pico" in TRUR_string.lower() or "vive" in TRUR_string.lower():    
        # 读取 JSON 文件
        print("-----------------", TRUR_string)
        with open(f'../lib/anchor_data/keyboard_layout_{TRUR_string}.json', 'r') as file:
            layout_data = json.load(file)
        keyboard_on_xy = []
        new_keyboard_name = []
        for key, value in layout_data.items():
            keyboard_on_xy.append([-value[0], value[1], 0])
            new_keyboard_name.append(key)
            if "enter" in key.lower():
                enter_key_index = len(new_keyboard_name) -1
        keyboard_on_xy = np.array(keyboard_on_xy) 
        key_name = new_keyboard_name
        ALL_intersection_point_list = [[keypoint[0], -keypoint[1], 0] for keypoint in ALL_intersection_point_list]

        scale = scale
        TRUR_string = TRUR_string
        anchor_x = ALL_intersection_point_list[-1][0]  - keyboard_on_xy[enter_key_index][0]*scale  
        anchor_y = ALL_intersection_point_list[-1][1]*scale  


    else:
        project_viewMatrix_inv = get_anchor_viewMatrix([0,0,0], -62-90, 1)
        ALL_intersection_point_list = np.array([global_to_local_point(keypoint[:3], project_viewMatrix_inv) for keypoint in ALL_intersection_point_list])
        ALL_intersection_point_list = np.array([rotate_points_around_axie(np.append(keypoint,1), 180, axis='y') for keypoint in ALL_intersection_point_list])
        keyboard_on_xy = np.load('../lib/anchor_data/keyboard_on_xy.npy')
        key_name = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'delete', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Enter', 'shift', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', 'S9', '!123', '.com', 'Space', 'web', 'set', 'cancel']
        enter_key_index = 20

        scale = scale
        TRUR_string = TRUR_string
        anchor_x = ALL_intersection_point_list[-1][0]  - keyboard_on_xy[enter_key_index][0]*scale  
        anchor_y = ALL_intersection_point_list[-1][1]


    anchor_position = [anchor_x, anchor_y, 0]
    keyboard_project_viewMatrix = get_anchor_viewMatrix(anchor_position, 0, scale)
    keyboard_on_xy_local = np.array([global_to_local_point(keypoint[:3], keyboard_project_viewMatrix) for keypoint in keyboard_on_xy])

    
    key_dx = 0.5 * abs(keyboard_on_xy_local[0][0] - keyboard_on_xy_local[1][0])
    key_dy = 0.5 * abs(keyboard_on_xy_local[0][1] - keyboard_on_xy_local[11][1])

    
    keyboard_xy_2d = keyboard_on_xy_local[:, :2]
    input_points = np.array(ALL_intersection_point_list)[:, :2]

    # plot_figure(input_points, keyboard_on_xy_local, key_name)
    # print("--------", scale)
    # exit()

    best_results = optimize_position_by_hit_count_match_and_dist(keyboard_xy_2d, input_points, key_name, key_dx, key_dy, search_range=0.02, step=0.001)  
    merged_time_ranges = load_json_data("../lib/savedata/merged_time_ranges.json")
    time_values = list(merged_time_ranges.values())[:-1]  
    if len(time_values) == 0:
        return []
    # average_time = sum(time_values) / len(time_values)

    save_string_list = []
    save_string2 = ""
    num_key = 1
    for (hit_count, total_dist, matched_keys, [dx, dy]) in best_results:
        key_motify = ""
        for idx, key in matched_keys:   
            key_motify += key.lower() *num_key   
            save_string2 += f"{idx}: {key.lower()}, " * num_key
        if key_type == "PIN":
            key_motify = motify_if_PIN(key_motify)    
        if key_type == "password":
            key_motify = motify_if_password(key_motify)    
        if key_type == "word":
            key_motify = motify_if_word(key_motify)    
        if key_type == "31key":
            key_motify = motify_if_31key(key_motify)    
        if key_type == "website":
            key_motify = motify_if_website(key_motify)    
        save_string_list.append(key_motify)
    
    save_string = "".join(save_string_list)
    if key_type == "31key":
        accuracy_list = cal_accuracy_31key(save_string_list)
    elif key_type == "website":
        accuracy_list = cal_accuracy_website(TRUR_string, save_string_list)
    else:
        accuracy_list = cal_accuracy_password(TRUR_string,save_string_list)


    # print("--------",key_type, accuracy_list)
    # exit()

    return accuracy_list

def test3_main(wether_plot, key_type, scale, TRUR_string):

    if "pico" in TRUR_string.lower() or "vive" in TRUR_string.lower():
        print("-----------------", TRUR_string)
        project_anchor = np.array([0, 0, -1])  # 重新定义为 Z 平面
        project_normal_vector = np.array([0, 0, -1])
        head_data_dic = load_json_data("../lib/savedata/merged_stare_when_input.json")
        anchor_data = head_data_dic
        index_half = 5+10
    else:
        project_anchor = np.load('../lib/anchor_data/project_anchor.npy')  
        project_normal_vector = np.load('../lib/anchor_data/project_normal_vector.npy')
        head_data_dic = load_json_data("../lib/savedata/merged_stare_when_input.json")
        anchor_data = load_json_data("../lib/savedata/stare_when_call_keyboard.json")
        index_half = int(len(anchor_data['Head_orientation']) *0.3)

    Head_orientation = anchor_data['Head_orientation'][index_half]
    Head_position = anchor_data['Head_position'][index_half]
    anchor_viewMatrix = get_view_matrix_from_head_pose(Head_position, Head_orientation)

    result_list = main(head_data_dic, anchor_viewMatrix, project_anchor, project_normal_vector, wether_plot, key_type, scale, TRUR_string)

    return result_list




if __name__ == "__main__":
    test3_main(1, "31key")  