import yaml

from loader import *

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
    l = DataLoader(conf['inputPath'], conf['modelPath'])
    # l.show_cur_item()
    l.create_model()
    
if __name__ == '__main__':
    main()
