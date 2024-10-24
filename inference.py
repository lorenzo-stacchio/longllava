import argparse
import os
import sys
LongLlavadir = "/home/vrai/BaraldiLongLLava/longllava/"
sys.path.append(f"{LongLlavadir}/LongLLaVA/")
sys.path.append(f"{LongLlavadir}/LongLLaVA/llava")
sys.path.append(f"{LongLlavadir}/LongLLaVA/data")
sys.path.append(f"{LongLlavadir}/LongLLaVA/scripts")
sys.path.append(f"{LongLlavadir}/LongLLaVA/utils")
from cli import Chatbot
import warnings
warnings.filterwarnings("ignore")
from plot_caption import plot_caption

parser = argparse.ArgumentParser(description='Args of Data Preprocess')
parser.add_argument('--model_dir', default='./LongLLaVA/LongLLaVA-9B', type=str)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument("--patchStrategy", type=str, default='norm', )
args = parser.parse_args()

bot = Chatbot(args)

# exit()
# query = '''You are an expert remote senser that analyzes satellite images and provide description about their content. 
#         I will provide you with a series of picture, and you will describe carefully each of them.
#         You must identify the presence of all the following attributes: ["cars","bus","parking","road","containers", "grass", "fields","factories","sea"].
#         In doing this, scan all the image: top-left, top-center, top-right, center-right, center-center, center-right, bottom-left, bottom-center, bottom-right.
#         Per each area, return a list of elements you identified.
#         Return no other comments.'''
folder_path = f"{LongLlavadir}/test_images"
image_paths = [] # image or video path
dict_image_answer = {}

for filename in os.listdir(folder_path):
    image_paths = []
    if os.path.isfile(os.path.join(folder_path, filename)) and "depth" not in filename:
        image_paths.append(os.path.join(folder_path, filename))


    query = '''You are an expert remote senser that analyzes satellite images and provide description about their content. 
            I will provide you with a series of picture, and you will describe carefully each of them.
            You must identify the presence of objects that are included in the following: cars, bus ,parking ,road ,containers", "grass", "fields ,factories ,sea.
            You must not include objects that are not present in the image.
            Return the list of elements present in the image without other comments.'''

    num_files = len(image_paths)
    query += 'Do this for the following image' + ' '.join(['<image>'] * num_files)


    bot = Chatbot(args)
    output = bot.chat(query, image_paths)
    print("\n\nAnswer 1: ", output) # Prints the output of the model

    # style of https://arxiv.org/html/2406.03843v1

    query2 = f'''Now, based on the objects you identified, which you described as ({output}), 
                    try to classify what this place is, from a remote sensing perspective (example amount, but are not limited to,
                        airport, baseball field, basketball court, dam, expressway service area, expressway toll station; golf field, ground track field, harbor, stadium, 
                        storage tank, tennis court, train station, industrial area.).'''
                    
                    
    output2 = bot.chat(query2)

    print("\n\nAnswer 2: ", output2) # Prints the output of the model
    
    formatted_text = "Answer 1:" +  output + "\n <Reasoning> \n Answer 2:" + output2
    dict_image_answer[filename] = formatted_text

    plot_caption(image_paths[0],formatted_text)
print(dict_image_answer)

# '''
# bot = Chatbot(args)
# output = bot.chat(query, image_paths)
# print(output) # Prints the output of the model
# '''

# while True:
#     text = input('Insert text ("q" to exit): ')
#     if text.lower() in ['q', 'quit']:
#         exit()
#     text += ' ' + ' '.join(['<image>'] * num_files)

#     answer = bot.chat(images=image_paths, text=text)

#     images = None # already in the history

#     print()
#     print(f'GPT: {answer}')
#     print()