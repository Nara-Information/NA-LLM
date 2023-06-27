import os 

import yaml 

def get_config(configPath): 
    with open(configPath) as f:
        return yaml.load(f, yaml.Loader)
    
def set_cred(credPath) -> dict:
    if os.path.exists(credPath):
        with open(credPath) as f:
            return yaml.load(f, yaml.Loader)
    else:
        print(f"Could not find a credential file at {credPath}")
        print("Try making one?")
        creds = dict()
        if input("[Y|n]: ").lower() == 'n':
            return dict()
        else:
            print("Put your own key for OpenAPI (data.go.kr) for fetching data.")
            print("HINT: 'decoded' version seems to work better.")
            print("(Say 'no' to skip giving the key.)")
            usr_in = input("KEY: ")
            if usr_in.strip().lower() == 'no': 
                creds['OpenAPI']=None
            else:
                creds['OpenAPI']=usr_in.strip()
            print("Put your OpenAI API key for GPT-baed augmentation.")
            print("(Say 'no' to skip giving the key.)")
            usr_in = input("KEY: ")
            if usr_in.strip().lower() == 'no': 
                creds['OpenAI']=None
            else:
                creds['OpenAI']=usr_in.strip() 
            return creds 
