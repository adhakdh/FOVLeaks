import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from lib1 import test1_main
from lib2 import test2_main
from lib3  import test3_main
import os
import json

if __name__ == "__main__":  
    item_list = ["website"]

    
    Result_dict = {}
    for item_type in item_list:
        input_dir = f"Dataset_FOVLeaks"
        for usr_name in os.listdir(input_dir)[:1]:
            usr_dir = os.path.join(input_dir, usr_name)
            for type_name in os.listdir(usr_dir):
                Result_dict[type_name + "1"] = []
            

    for item_type in item_list:
        input_dir = f"Dataset_FOVLeaks"
        for usr_name in os.listdir(input_dir):
            usr_dir = os.path.join(input_dir, usr_name)
            for type_name in os.listdir(usr_dir):
                type_dir = os.path.join(usr_dir, type_name)
                for file_name in os.listdir(type_dir):
                    file_dir = os.path.join(type_dir, file_name)
                    scale = file_name.split(".txt")[0].split("_")[1]
                    scale = float(scale)
                    TRUR_string = type_name + ".com"
                    test1_main(file_dir,0)
                    test2_main()
                    result_list = test3_main(0, item_type, scale, TRUR_string)  
                    if not result_list:
                        print(file_dir)
                        
                    else:
                        Result_dict[type_name + "1"].append([file_dir] + result_list[:3])  


    
    file_name = f"../lib/savedata/result_{item_type}.json"
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(Result_dict, json_file, ensure_ascii=False, indent=4)