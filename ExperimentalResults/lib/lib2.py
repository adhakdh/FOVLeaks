from lib import merge_close_points

def test2_main():
    input_file = "../lib/savedata/stare_when_input.json"
    output_file = "../lib/savedata/merged_stare_when_input.json"
    time_log_path="../lib/savedata/merged_time_ranges.json"
    merge_close_points(input_file, output_file, time_log_path)




if __name__ == "__main__":
    test2_main()