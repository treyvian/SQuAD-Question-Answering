import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_json_file", help="Path to the json file", type=str)
    args = parser.parse_args()