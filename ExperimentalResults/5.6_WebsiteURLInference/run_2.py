import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
import json
from lib import cal_file_accuracy

if __name__ == "__main__":
    item_list = ["website"]  
    for item_type in item_list:
        file_name = f"../lib/savedata/result_{item_type}.json"
        Result_dict = json.load(open(file_name, 'r', encoding='utf-8'))
        top1_result_dict, top3_result_dict, tmp_len_list = cal_file_accuracy(Result_dict)

        len_dict = len(top1_result_dict.keys())
        top1_accuracy = sum(top1_result_dict.values()) / len_dict
        top3_accuracy = sum(top3_result_dict.values()) / len_dict
        valid_key = sum([i[0] for i in tmp_len_list])
        all_key = sum([i[1] for i in tmp_len_list])
        print(f"Result === type: {item_type} == key_num：{valid_key}/{all_key} == top1:{top1_accuracy:<.3f}  top3:{top3_accuracy:<.3f}")

        
        file_name = f"../lib/savedata/cal_{item_type}.json"
        merged_result_dict = {key: [top1_result_dict[key], top3_result_dict[key]] for key in top1_result_dict}
        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(merged_result_dict, json_file, ensure_ascii=False, indent=4)

        for key, value in top1_result_dict.items():
            print(f"{key:<20}: ----- {len(Result_dict[key]):<2} ----- {top1_result_dict[key]:.3f} ----- {top3_result_dict[key]:.3f}")
        print("\n")
