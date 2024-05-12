import whisper
from transformers import WhisperForConditionalGeneration
import torch
from tqdm import tqdm

# using pickle to serialize the map_dict
import pickle

# to enable verbose printing of exceptions (+ layers matching name)
DEBUG = False

# set to True if your custom model has been trained using DDP (multi-gpu)
# as in my case, in the custom HF model, keys have a prefix (model.)
# it should come from the fact that I have trained on a milti-gpu machine, using DDP
DDP_TRAINED = True

# if DDP we have to add a prefix to match with the HF state_dict
if DDP_TRAINED:
    PREFIX = "model."
else:
    PREFIX = ""

# for now, tested only with medium
MODEL_SIZE = "medium"

# the device where you're running this code
DEVICE = "cpu"

# the name of the file with your fine-tuned model
FINE_TUNED_MODEL = "medium-custom.pt"


# the name of the file for the serialized map_dict
# a different name, to avoid overwrite it
FILE_DICT = "map_dict_test.pkl"


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


# next functions are used to make sanity checks for the mappings


# get if it is encoder or decoder
def extract_function(key_name):
    # encoder or decoder is the first part of the key
    first_part = key_name.split(".")[0]

    key_func = None
    if first_part in ["enconder", "decoder"]:
        key_func = first_part

    return key_func


def extract_layer_num(key_name):
    # layer num is the third piece
    layer_num = None

    if has_numbers(key_name):
        layer_num = key_name.split(".")[2]

    return layer_num


# check that the two keys are for layers
# with the same function
# (both encoder or both decoder)
# and have the same layer number
# this way we are super-safe (I think)
def sanity_check(key1, key2):
    is_ok = True

    # check same func (encoder or decoder)
    func1 = extract_function(key1)
    func2 = extract_function(key2)

    if func1 != func2:
        print(
            f"Warning: layers seem to have different functions: {key1},{key2}"
        )
        is_ok = False

    # check same layer_num
    layer1 = extract_layer_num(key1)
    layer2 = extract_layer_num(key2)

    if layer1 != layer2:
        print(f"Warning: layers seem to have different numbers: {key1},{key2}")
        is_ok = False

    return is_ok


# Vanilla means: not custom trained
print()
print("Loading vanilla Whisper model")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

print("Loading vanilla HF Model")
hugging_face_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-" + MODEL_SIZE
)

# extract state-dict from both
state_d_openai = model.state_dict()
state_d_huggingface = hugging_face_model.state_dict()

# build the mapping between keys...
map_dict = {}
print("Matching layers...")

# for every layer in OpenAI model
n_sanity_ok = 0

#
# here we're considering the cartesian product of the two state dict and try to match
# rules applied:
# 1. the two layers have the same shape
# 2. the two layer have the same parameters' values
# 3. we apply sanity check (see function above)
#
for k in tqdm(state_d_openai):
    # find a layer in the HF model, check with j
    for j in state_d_huggingface:
        # where parameters have same shape and same values
        if state_d_huggingface[j].shape == state_d_openai[k].shape:
            if torch.all(
                torch.eq(state_d_huggingface[j], state_d_openai[k])
            ).item():
                # found, register the mapping
                map_dict[k] = j
                # make some check and eventually print a warning
                if sanity_check(k, j) == True:
                    n_sanity_ok += 1

                    # if you enable thsi print you can see the name of the layer
                    # chosen in the match and you will se that they have the same functions
                    if DEBUG:
                        print(k, j)

                break

# check if we have matched every entry
print("Check if we have matched every entry in state_dict...")
print()
print(f"Number of keys: {len(map_dict.keys())}")
assert len(map_dict.keys()) == len(
    state_d_openai.keys()
), "The match is not complete !"

print(f"Number of sanity_check ok: {n_sanity_ok}")
print()

print("Match is complete !!!")
print()

# serialize the map_dict to file
print("Serializing map_dict...")

with open(FILE_DICT, "wb") as f:
    pickle.dump(map_dict, f)
    f.close()

print(f"map_dict saved as: {FILE_DICT}...")
print()

# loading with match keys
# restart from pickle file
print("Reloading map_dict...")
print()
with open(FILE_DICT, "rb") as f:
    map_dict = pickle.load(f)

# loading fine-tuned dict
print("Loading fine tuned dict...")
# added map_location to handle the fact that the custom model has been trained on GPU
state_dict_finetuned = torch.load(
    FINE_TUNED_MODEL, map_location=torch.device(DEVICE)
)

# build the state_dict to be used
# take the key name from standard (OpenAI) and the value from finetuned (HF)
print("Rebuild the state dict...")
new_state_dict = {}
n_except = 0
for k in tqdm(map_dict.keys()):
    try:
        # You must add "model." if you have used DDP in custom training
        # see DDP_TRAINED above
        # PREFIX is added to a HF fine-tuned 8with DDP). It is not in vanulla HF models
        new_state_dict[k] = state_dict_finetuned[PREFIX + map_dict[k]]
    except:
        n_except += 1

        if DEBUG:
            print(PREFIX + map_dict[k])

msg_err = f"Rebuild state dict failed, {n_except} pick failed"
assert n_except == 0, msg_err

print()
print("Loading the final model...")
model.load_state_dict(new_state_dict)
