import numpy as np
import os
from lib import read_head_data, compute_derivative, save_stare_when_call_keyboard, save_stare_when_input, detect_events, save_events

def test1_main(file_path, wether_plot=0):
    head_data_dic = read_head_data(file_path)     
    times = head_data_dic["IMUTime"]
    if "pico" in file_path.lower() or "vive" in file_path.lower():
        print("-----------------", file_path)
        z_dirs = -np.array(head_data_dic["Head_Z_direction"])
    else:
        z_dirs = np.array(head_data_dic["Head_Z_direction"])
    derivative, _ = compute_derivative(z_dirs, times)      
    events, dy_dt = detect_events(z_dirs, derivative, times)
    try:
        save_events(head_data_dic, events)
        save_stare_when_input(head_data_dic, events)    
        save_stare_when_call_keyboard(head_data_dic, events)
    except:
        raise ValueError("error.")
    if wether_plot:
        pass




if __name__ == "__main__":
    input_dir = "test4"
    file_name_list = os.listdir(input_dir)
    sorted_file_list = sorted(file_name_list, key=lambda x: int(os.path.splitext(x)[0]))
    file_list = [ os.path.join(input_dir, item)  for item in sorted_file_list]
    file_path = file_list[-1]
    test1_main(file_path)
    print("------------", file_path)