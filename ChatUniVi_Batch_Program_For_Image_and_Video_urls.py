
from PIL import Image
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query
import torch
import os
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
from pytube import YouTube
import requests
import csv
import cv2
import pandas as pd
import re
from fastapi.middleware.cors import CORSMiddleware
model_path = "Chat-UniVi/Chat-UniVi"  # or "Chat-UniVi/Chat-UniVi-13B"
conv_mode = "simple"
temperature = 0.2
top_p = None
num_beams = 1

disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = "ChatUniVi"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

def predict_image(image_url,qs):
    print(image_url)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    # Check if the video exists
    if image_url is not None:
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image_url, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        return outputs


def download_youtube(video_url): #function to downoad youtube
    VIDEO_SAVE_DIRECTORY = "/home/azureuser/ChatUnivi2/Chat-UniVi/Main Videos/Youtube Videos"
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




def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

from utilsfuntion import *
def predict_video(video_path_or_link, prompt):
    try:

        # if "tiktok.com" in video_path_or_link:
        #     path=get_tiktok_video(video_path_or_link)
        #     print(path)
        # elif "instagram.com" in video_path_or_link:
        #     link,vision_type=from_rapid_api_Video_download_link_from_instagaram(url)
        #     if vision_type=='video':
        #         path=download_instagram_video_to_mp4(link)
        #     elif vision_type=='image':
                
        #     path = download_instagram(video_path_or_link)
        # elif "youtube.com" in video_path_or_link:
        #     path = download_youtube(video_path_or_link)
        # else:
        #     path = video_path_or_link


        video_path = video_path_or_link

        # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
        # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
        max_frames = 2000

        # The number of frames retained per second in the video.
        video_framerate = 1

        if prompt is None: # Input Text
            qs = "Describe the video."
        else:
            qs = prompt

        # Sampling Parameter
        conv_mode = "simple"
        temperature =0.2
        top_p = None
        num_beams = 1
        
        # "default": simple_conv,
        # "simple": simple_conv,
        # "simpleqa": simple_qa,
        # "v1": conv_v1,
        results=[video_path_or_link,prompt,video_path]


        

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

        if model.config.config["use_cluster"]:
            for n, m in model.named_modules():
                m = m.to(dtype=torch.bfloat16)

        # Check if the video exists
        if video_path is not None:
            video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=max_frames, video_framerate=video_framerate)

            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video_frames.half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output_ids = output_ids.sequences
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            results.append(outputs)
            # Placeholder for model processing logic
            generated_response = f"Processed video from {video_path_or_link} with input text: {prompt}"

            print(results)
            # return {"response": outputs}
            return outputs
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def Driver(url,key):
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

    if key.lower()=="haix":
        if "tiktok.com" in url:
            link=from_rapid_api_Video_download_link_from_tiktok(url)
            path=download_tiktok_video_to_mp4(link)
            print(path)
            return videoPrompt,predict_video(path,videoPrompt)
        elif "instagram.com" in url:
            link,vision_type=from_rapid_api_Video_download_link_from_instagaram(url)
            print(vision_type)
            
            if vision_type=='video':
                
                path=download_instagram_video_to_mp4(link)
                print(path)
                return videoPrompt,predict_video(path,videoPrompt)
            elif vision_type=='image':
            #    link,_ = from_rapid_api_Video_download_link_from_instagaram(url)
                print(link)
                path=download_instagram_video_to_jpg(link)
                import cv2
                img=cv2.imread(path)
                print(path)
                return Imageprompt,predict_image(img,Imageprompt)
        elif "youtube.com" in url:
            path = download_youtube(url)
            return videoPrompt,predict_video(path,videoPrompt)
    else:
        return "Key error"
file='/home/azureuser/ChatUnivi2/Chat-UniVi/Input_url_folder/Input_urls_for_batch_program.txt'


with open(file,'r') as fp:
    links = fp.readlines()


Data={
    "Link":[],
    "Prompt":[],
    "ChatUnivi Output":[]
}

print(links)
for url in links:
    print(url)
    url=url.replace('\n','')
    output=Driver(url,'haix')
    Data['Link']+=[url]
    Data['Prompt']+=[output[0]]
    Data['ChatUnivi Output']+=[output[1]]

# 
import datetime
data=pd.DataFrame(Data)
name=file.split('/')[-1].split('.')[0]
file_location=f'/home/azureuser/Results/results_{datetime.datetime.now()}_{name}.csv'
data.to_csv(file_location,index=False)


