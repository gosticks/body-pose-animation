from loader import *
import yaml

ascii_logo = """\
  /$$$$$$  /$$      /$$ /$$$$$$$  /$$   /$$     /$$
 /$$__  $$| $$$    /$$$| $$__  $$| $$  |  $$   /$$/
| $$  \__/| $$$$  /$$$$| $$  \ $$| $$   \  $$ /$$/ 
|  $$$$$$ | $$ $$/$$ $$| $$$$$$$/| $$    \  $$$$/  
 \____  $$| $$  $$$| $$| $$____/ | $$     \  $$/   
 /$$  \ $$| $$\  $ | $$| $$      | $$      | $$    
|  $$$$$$/| $$ \/  | $$| $$      | $$$$$$$$| $$    
 \______/ |__/     |__/|__/      |________/|__/    
                                                   
"""

def load_config():
    with open('./config.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def main():
    print(ascii_logo)
    conf = load_config()
    print("config loaded")
    l = DataLoader(conf['inputPath'])
    l.show_cur_item()
    l.load_keypoints()
    
if __name__ == '__main__':
    main()