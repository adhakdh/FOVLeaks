import json
import numpy as np


item_type = "31key"
file_name = f"../lib/savedata/result_{item_type}.json"
Result_dict = json.load(open(file_name, 'r', encoding='utf-8'))


true_key_list = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 
                '1', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', 
                '2', '3']
replacement_map = {
    'shift': '1', 
    'space': '2', 
    'enter': '3',
}

top1_result_dict = {key: [] for key in true_key_list}
top3_result_dict = {key: [] for key in true_key_list}


tmp_len_list = []
for file_data in Result_dict[item_type]:     
    tmp_accuracy = file_data[1][0]   
    if tmp_accuracy < 0.71:
        continue
    top1_string = file_data[1][1]
    top3_string = ''.join(key[-1] for key in file_data[1:]) 
    for tmp_key,value in replacement_map.items(): 
        top1_string = top1_string.replace(tmp_key, value)
        top3_string = top3_string.replace(tmp_key, value)
    tmp_len = min(len(top1_string), 31) + 1 
    tmp_len2 = 31 + 1 
    tmp_len_list.append([tmp_len,tmp_len2])
    for key in true_key_list:
        top1_result_dict[key].append(1 if key in top1_string else 0)
        top3_result_dict[key].append(1 if key in top3_string else 0)

len_dict = len(top1_result_dict["q"])
top1_result_dict = {key: sum(top1_result_dict[key]) / len_dict for key in true_key_list}
top3_result_dict = {key: sum(top3_result_dict[key]) / len_dict for key in true_key_list}

top1_accuracy = sum(top1_result_dict.values()) / len(true_key_list)
top3_accuracy = sum(top3_result_dict.values()) / len(true_key_list)
valid_key = sum([i[0] for i in tmp_len_list])
all_key = sum([i[1] for i in tmp_len_list])
print(f"Result === type: {item_type} == key_num：{valid_key}/{all_key} == top1:{top1_accuracy:<.3f}  top3:{top3_accuracy:<.3f}")

for key in true_key_list:
    print(f"{key:<20}: ----- {len_dict:<2} ----- {top1_result_dict[key]:.3f} ----- {top3_result_dict[key]:.3f}")