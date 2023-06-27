import sys 

from train import causal, translate
from utils.configs import get_config, set_cred 
from utils.data.augment import augment
from utils.data.construct_data import construct_data

MODES = (
    ('fetch', 'Fetch the data from the OepnAPI service'),
    ('augment', 'Augment the fetched data with OpenAI API'),
    ('trainTranslate', 'Perform fine-tuning with translate (i.e., Sequence-to-Sequence) objective'),
    ('trainCausal', 'Perform fine-tuning with causal LM objective'),
)

def _get_mode(argv):
    modes = [i[0].lower() for i in MODES]
    if len(argv) > 1:
        if argv[1] in modes: return MODES[modes.index(argv[1])][0]
    while True:
        print("Choose one of the supported modes: ")
        for i, mode in enumerate(MODES):
            print(f"{i+1}\t{mode[0]:20}\t\t{mode[1]}")
        choice = input("Put the number for the mode: ")
        try:
            if 1 <= int(choice) <= 4: 
                return MODES[int(choice)-1][0]
        except ValueError:
            pass
        print("Got an invalid mode.\n")

def main():
    mode = _get_mode(sys.argv)

    args = get_config('config.yaml')
    creds = set_cred(args['credPath'])

    if mode == MODES[0][0]: # fetch
        construct_data(args['fetching'], creds)
    elif mode == MODES[1][0]: # augment
        augment(args['augmenting'], creds)
    elif mode == MODES[2][0]: # trainTranslate
        translate.train(args)
    elif mode == MODES[3][0]: # trainCausal
        causal.train(args)
    
if __name__ == "__main__":
    main()
