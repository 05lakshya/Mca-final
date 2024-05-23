import torch
import pandas as pd
import time
import os
import requests
import utilsfuntion
from fastapi import FastAPI, HTTPException, Query

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from pytube import YouTube
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware
model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = 'cache_dir'

device = 'cuda'
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)

import os
def get_tiktok_video(url):
    endpoint_url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/tiktok"
    headers = {
	"X-RapidAPI-Key": "38fb340264mshc8e12e648c185b2p117435jsnf23d4d503b96",
	"X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
}

    query_params = {
        "url": url
    }
    video_url=url.replace('/','')
    folder_path='/home/azureuser/Video-LLAVA-2/Main Videos/Tik Tok VIdeos'
    globall='/home/azureuser/Main Videos/Instagram Videos'

    if video_url not in os.listdir(folder_path):
        try:
            response=requests.get(endpoint_url,headers=headers,params=query_params)

            video =requests.get(response.json()['links'][0]['link']).content
            with open(f'/home/azureuser/Video-LLAVA-2/Main Videos/Tik Tok VIdeos/{video_url}.mp4','wb') as fp:
                fp.write(video)

            with open(os.path.join(globall,video_url+'.mp4'),'wb') as fp:
                fp.write(video)
            

            return os.path.join(folder_path,video_url+'.mp4')
        except:
            return "Video Not found"
        
    else:
        return os.path.join(folder_path,video_url+'.mp4')

def download_instagram(video_url):
  local='/home/azureuser/Video-LLAVA-2/Main Videos/Instagram Videos'
  globall='/home/azureuser/Main Videos/Instagram Videos'
  url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/instagram"
  querystring = {"url": video_url}
  headers = {
	"X-RapidAPI-Key": "38fb340264mshc8e12e648c185b2p117435jsnf23d4d503b96",
	"X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
}
  video_url=video_url.replace('/','')
  if video_url+'.mp4' not in os.listdir(local):
    print(video_url not in os.listdir(local))
    try:
      response=requests.get(url,headers=headers,params=querystring)
    #   print(response.json())
      video=requests.get(response.json()['links'][0]['link']).content
      with open(os.path.join(globall,video_url+'.mp4'),'wb') as fp:
        fp.write(video)

      with open(os.path.join(local,video_url+'.mp4'),'wb') as fp:
        fp.write(video)
      return os.path.join(local,video_url+'.mp4')
    except Exception as e:
      return f'exception ouccured{e}'
  else:
    print('video found in local')
    return os.path.join(local,video_url+'.mp4')



def download_youtube(video_url): #function to downoad youtube
    VIDEO_SAVE_DIRECTORY = "/home/azureuser/Video-LLAVA-2/Main Videos/Youtube Videos"
    video_id = YouTube(video_url).video_id
    video_path = os.path.join(VIDEO_SAVE_DIRECTORY, f'{video_url}.mp4')
    
    # Check if video already exists
    # if os.path.exists(video_path):
        # print("Video already exists in the directory.")
        # return os.path.relpath(video_path)  # Return relative path to existing video
    
    video = YouTube(video_url)
    video = video.streams.get_highest_resolution()

    try:
        video_path = video.download(VIDEO_SAVE_DIRECTORY)
        print(f"Video downloaded successfully as {video_path}.")
        return os.path.relpath(video_path)  # Return relative path to newly saved video
    except Exception as e:
        print(f"Failed to download video due to {e}")
        return None  # Return None if download fails
def predict_image(url,inp):
    image_processor=processor['image']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # image=Image.open(BytesIO(requests.get(url).content))
    # getImage(url)
    image_tensor = image_processor.preprocess(url, return_tensors='pt')['pixel_values']
    if type(image_tensor) is list:
        tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        tensor = image_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    return outputs

def predict_video(video,inp):
    

    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    tokenizer, model, processor
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    return outputs
from utilsfuntion import *
# # http://51.105.246.1:5000/VideoLLaVAImageVideo/?url=<urlhere>&VisionType=<vision type here image or video>&Prompt=<prompt here>
# app=FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow requests from any origin
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["Authorization", "Content-Type"],
# )
# @app.get('/api/videolva_vision/')
def Driver(url):
    videoPrompt="""
Thoroughly check the video content by analysing all individual frames, and any accompanying text to identify up to 5 prominent keywords, up to 5 key entities,
 and up to 5 significant topics depicted or mentioned that convey the message in the video. Ensure your analysis is solely based on the provided information, avoiding assumptions , 
 repetition, irrelevant, additional, extraneous details. Present your findings in a comma-separated format for each category in a new line, as follows and DO NOT REPEAT THE WORDS IN EACH CATEGORY:\n "Top Keywords": {List the most essential words or phrases crucial for understanding the content and message of all the video frames. Ensure the response for Top Keywords is formatted with comma separated values on a single line like, Top Keywords: (example 1), (example 2), (example 3), (example 4), (example 5).} \n "Visible Entities" : Identify only the main entities that are clearly visible or directly mentioned in the video frames (do not include entities in the response that are not clearly visible in video frames). Ensure no assumptions are made about generic products and entities. Include entities in response that are explicitly recognizable through logos, text or direct mentions in the video. List upto top 5 visible entities from the video using only single words or phrases, without providing any additional information. Ensure the response for  Visible Entities is formatted with comma separated values on a single line like, Visible Entities/Brands: (example 1), (example 2), (example 3), (example 4), (example 5).} \n "Significant Topics" : {Highlight significant topics or themes depicted or referenced in the video frames and any accompanying text. Ensure the response for Significant Topics is formatted with comma separated values on a single line like, Significant Topics : (example 1), (example 2), (example 3), (example 4), (example 5).} \n "Video Summary" : {Provide a summary of the video up to 50 words, focusing on its main theme and key scenes, and include relevant text. 
Make sure the summary is precise, capturing the most significant aspects of the video frames and leaving out irrelevant details.}
"""
    Imageprompt="""
"Analyze the image and accompanying text to provide summary and identify up to 5 prominent keywords, up to 5 key brand names/entities, and 5 significant topics depicted or mentioned that convey the message in the picture. Ensure your analysis is solely based on the provided information, avoiding assumptions , repetition, irrelevant, additional, extraneous details. Present your findings in a comma-separated format, with separate lines for each category, as follows: (if no relevant information found, keep it blank for that category). Do not repeat the words in each category.
"Top Keywords": {List the most essential words or phrases crucial for understanding the content and message of the picture. Ensure the response for Top Keywords is formatted with comma separated values on a single line like, Top Keywords:  example 1, example 2, example 3, example 4, example 5.}/n
"Visible Entities/objects Identified": {Identify the objects, entities, visible in the picture or mentioned in the accompanying text. Avoid assuming generic products as specific ; only include  that are explicitly recognizable in the video content.  Ensure the response for this is formatted with comma separated values on a single line like, Visible Brands/Objects Identified: example 1, example 2, example 3, example 4, example 5.} /n
"Significant Topics": {Highlight significant topics or themes depicted or referenced in the picture and accompanying text. Ensure the response for this is formatted with comma separated values on a single line like, Significant Topics: example 1, example 2, example 3, example 4, example 5.}/n
"Image Summary": {Summarize the scene depicted in the picture along with any accompanying text, focusing on the central theme, message conveyed, including elements such as the setting, main subjects, and any noteworthy features. Limit words if needed}"

"""
    key='haix'
    if key.lower()=="haix":
        if "tiktok.com" in url:
            link=from_rapid_api_Video_download_link_from_tiktok(url)
            path=download_tiktok_video_to_mp4(link)
            print(path)
            return predict_video(path,videoPrompt)
        elif "instagram.com" in url:
            link,vision_type=from_rapid_api_Video_download_link_from_instagaram(url)
            print(vision_type)
            
            if vision_type=='video':
                
                path=download_instagram_video_to_mp4(link)
                print(path)
                return predict_video(path,videoPrompt)
            elif vision_type=='image':
            #    link,_ = from_rapid_api_Video_download_link_from_instagaram(url)
                print(link)
                path=download_instagram_video_to_jpg(link)
                import cv2
                img=cv2.imread(path)
                print(path)
                return predict_image(img,Imageprompt)
        elif "youtube.com" in url:
            path = download_youtube(url)
            return predict_video(path,videoPrompt)
    else:
        return "Key error"
import argparse
# Make Sure use the same result file which you used for chatUNIVI
filename='/home/azureuser/Results/results_2024-03-03 15:06:08.572637.csv'
data=pd.read_csv(filename)
data['Video LLaVA']=data['Link'].apply(Driver)


data.to_csv(filename,index=False)
print(data.head())

from pyngrok import ngrok

    ngrok.set_auth_token('2fajI0Fk3xuXCICMrLi74NMoArz_34RiLCugo6J3QtrJwadJG')
    # from flask import Flask,request
    # from flask_ngrok import run_with_ngrok
    ngrok_tunnel = ngrok.connect(5000)
    print("Public URL:", ngrok_tunnel.public_url)
    app.run()