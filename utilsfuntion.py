import os
import base64
from PIL import Image
from io import BytesIO
import requests
from pytube import YouTube
import datetime
from datetime import timedelta
import time
def load_image(image_file):
    """
    Function to decode a base64 image string, save it to a temporary folder, and upload it to GitHub.
    
    Parameters:
    - image_file (str): Base64 encoded image string.

    Returns:
    - image (PIL.Image): The image loaded from the base64 string.
    - image_path (str): File path where the image is saved.
    """
    
    # Decode the base64 image string to raw image data
    image_data = base64.b64decode(image_file)
    
    # Open the raw image data using PIL's Image library
    raw_image = Image.open(BytesIO(image_data))
    
    # Define the folder path where the image will be saved
    temp_folder_path = "/home/azureuser/temp_folder"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    
    # Define the path for saving the image
    image_filename = "temp_image.jpg"
    image_path = os.path.join(temp_folder_path, image_filename)
    
    # Save the image to the specified folder path
    raw_image.save(image_path)
    
    # Convert the saved image to RGB format
    image = Image.open(image_path).convert('RGB')
    
    # Return the loaded image and its file path
    return image, image_path
def from_rapid_api_Video_download_link_from_tiktok(urllink):
    """
    Download video from Instagram using RapidAPI.

    Parameters:
    - urllink (str): Instagram video URL.

    Returns:
    - output (str): Video download link obtained from RapidAPI.

    Note:
    - The function tries two different endpoints from RapidAPI to fetch the video download link.
      If the first attempt fails, it falls back to the second endpoint.
    - If both attempts fail, it prints a message indicating the failure and returns None.
    """

    try:
        # First Attempt: Using 'tiktok-video-no-watermark2' endpoint
        url = "https://tiktok-video-no-watermark2.p.rapidapi.com/"
        querystring = {"url": urllink, "hd": "1"}
        headers = {
            "X-RapidAPI-Key": "4b441a77c2msh03f434a47f159eap143497jsn6810dff6c407",
            "X-RapidAPI-Host": "tiktok-video-no-watermark2.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring).json()
        output = response['data']['wmplay']
        return output
    except Exception as e1:
        try:
            # Second Attempt: Using 'social-media-video-downloader' endpoint
            url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/tiktok"
            querystring = {"url": urllink}
            headers = {
                "X-RapidAPI-Key": "4b441a77c2msh03f434a47f159eap143497jsn6810dff6c407",
                "X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
            }
            response = requests.get(url, headers=headers, params=querystring).json()
            output = response['links'][0]['link']
            return output
        except Exception as e2:
            print(f"For the video link {urllink}, we are unable to generate it. Error: {e1}, {e2}")
            return None
def from_rapid_api_Video_download_link_from_instagaram(urllink):
    """
    Get video download link from Instagram using RapidAPI.

    Parameters:clea
    - urllink (str): Instagaram video URL.

    Returns:
    - output (str): Video download link obtained from RapidAPI.

    Note:
    - The function tries two different endpoints from RapidAPI to fetch the video download link.
      If the first attempt fails, it falls back to the second endpoint.
    - If both attempts fail, it prints a message indicating the failure and returns None.
    """
    if "/reel/" in urllink:
        shortcode=urllink[31:-1]
    elif "/p/" in urllink:
        shortcode=urllink[28:-1]
    elif "/tv/" in urllink:
        shortcode=urllink[29:-1]
    url = f"https://instagram-api-20231.p.rapidapi.com/api/media_info_from_shortcode/{shortcode}"
    headers = {
            "X-RapidAPI-Key":  "4b441a77c2msh03f434a47f159eap143497jsn6810dff6c407",
            "X-RapidAPI-Host": "instagram-api-20231.p.rapidapi.com"
        }
    try:
        response = requests.request("GET", url, headers=headers).json()
        if "video_versions" in response['data']['items'][0]:
            output = response['data']['items'][0]['video_versions'][0]['url']
            content_type='video'
            return output,content_type
        else:
            output = response['data']['items'][0]['image_versions2']['candidates'][0]['url']
            content_type='image'
            return output,content_type
            

        
    except:
        try:
            # If the first attempt fails, make a second attempt with a delay
            response = requests.request("GET", url, headers=headers).json()
            time.sleep(1.5)
            if "video_versions" in response['data']['items'][0]:
                output = response['data']['items'][0]['video_versions'][0]['url']
                ontent_type='video'
                return output,content_type
            else:
                output = response['data']['items'][0]['image_versions2'][4]['url']
                content_type='image'
                return output,content_type
            
        except Exception as e1:
            try:
                # Second Attempt: Using 'social-media-video-downloader' endpoint
                url = "https://social-media-video-downloader.p.rapidapi.com/smvd/get/tiktok"
                querystring = {"url": urllink}
                headers = {
                    "X-RapidAPI-Key": "4b441a77c2msh03f434a47f159eap143497jsn6810dff6c407",
                    "X-RapidAPI-Host": "social-media-video-downloader.p.rapidapi.com"
                }
                response = requests.get(url, headers=headers, params=querystring).json()
                output = response['links'][0]['link']
                return output
            except Exception as e2:
                print(f"For the video link {urllink}, we are unable to generate it. Error: {e1}, {e2}")
                return None

def download_tiktok_video_to_mp4(video_url, filename="tiktok_downloaded_video.mp4"):
    """
    Download TikTok video from a given URL and save it as an MP4 file.

    Parameters:
    - video_url (str): URL of the TikTok video.
    - filename (str): Name of the file to save the downloaded video (default is 'tiktok_downloaded_video.mp4').

    Returns:
    - file_path (str): Relative path of the downloaded video file.

    Note:
    - The function creates a folder ('tiktok_videos') if it doesn't exist and saves the video in that folder.
    - It makes two attempts to get the video content from the URL using the 'requests' library.
    - If the download is successful, it returns the relative path of the downloaded video file.
    - If the download fails, it prints an error message and returns None.
    """

    # Create a folder if it doesn't exist
    folder_path = 'tiktok_videos'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path within the folder
    file_path = os.path.join(folder_path, filename)

    # Initialize the response variable to None
    response = None

    try:
        # Attempt to get the video content from the URL
        response = requests.get(video_url, stream=True)
    except Exception as e1:
        try:
            # Retry the download in case of the first attempt failure
            response = requests.get(video_url, stream=True)
        except Exception as e2:
            pass

    if response and response.status_code == 200:
        # Save the downloaded video content to the specified file path
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Video downloaded successfully as {file_path}.")
        return file_path  # Return relative path
    else:
        print("Failed to download the video.")
        return None  # Return None if download fails


def download_instagram_video_to_mp4(video_url):
    """
    Download a video from an Instagram post using the provided URL.

    Parameters:
    - video_url (str): URL of the Instagram video post.

    Returns:
    - video_path (str): Relative path of the downloaded video file.

    Note:
    - The function creates a folder ('instagram_videos') if it doesn't exist and saves the video in that folder.
    - It attempts to download the video content from the Instagram URL using the 'requests' library.
    - If the download is successful, it saves the video in the 'instagram_videos' folder and returns the relative path.
    - If the download fails, it prints an error message and returns None.
    """

    # Create a folder if it doesn't exist
    if not os.path.exists('instagram_videos'):
        os.makedirs('instagram_videos')

    try:
        # Attempt to get the video content from the Instagram URL
        video_response = requests.get(video_url)

        if video_response.status_code == 200:
            # Create a directory if it doesn't exist
            video_path = os.path.join('instagram_videos', f'instagram_downloaded_video.mp4')

            # Save the downloaded video content to the specified file path
            with open(video_path, "wb") as f:
                f.write(video_response.content)

            print(f"Video downloaded successfully as {video_path}.")
            return video_path  # Return the path where the video is saved
        else:
            print("Failed to download video.")
            return None
    except Exception as e:
        print("Failed to download video.")
        return None
    
def download_instagram_video_to_jpg(image_url):
    """
    Download a video from an Instagram post using the provided URL.

    Parameters:
    - video_url (str): URL of the Instagram video post.

    Returns:
    - video_path (str): Relative path of the downloaded video file.

    Note:
    - The function creates a folder ('instagram_videos') if it doesn't exist and saves the video in that folder.
    - It attempts to download the video content from the Instagram URL using the 'requests' library.
    - If the download is successful, it saves the video in the 'instagram_videos' folder and returns the relative path.
    - If the download fails, it prints an error message and returns None.
    """

    # Create a folder if it doesn't exist
    if not os.path.exists('instagram_images'):
        os.makedirs('instagram_images')

    try:
        # Attempt to get the video content from the Instagram URL
        image_response = requests.get(image_url)

        if image_response.status_code == 200:
            # Create a directory if it doesn't exist
            image_path = os.path.join('instagram_images', f'instagram_downloaded_image.jpg')

            # Save the downloaded video content to the specified file path
            with open(image_path, "wb") as f:
                f.write(image_response.content)

            print(f"Image downloaded successfully as {image_path}.")
            return image_path  # Return the path where the video is saved
        else:
            print("Failed to download Image.")
            return None
    except Exception as e:
        print("Failed to download Image.")
        return None

def download_youtube(video_url):
    """
    Download a YouTube video using the provided URL.

    Parameters:
    - video_url (str): URL of the YouTube video.

    Returns:
    - video_path (str): Relative path of the downloaded video file.

    Note:
    - The function specifies a directory ("/home/azureuser/youtube_videos") to save downloaded YouTube videos.
    - It extracts the video ID from the YouTube URL and defines the filename as "youtube_downloaded_video.mp4".
    - It checks if the video already exists in the specified directory; if so, it returns the relative path to the existing video.
    - If the video does not exist, it attempts to download the highest resolution stream of the YouTube video.
    - If the download is successful, it returns the relative path to the newly saved video.
    - If the download fails, it prints an error message and returns None.
    """

    # Directory to save downloaded YouTube videos
    VIDEO_SAVE_DIRECTORY = "/home/azureuser/youtube_videos"

    # Extract video ID and define the filename
    video_id = YouTube(video_url).video_id
    video_filename = f"youtube_downloaded_video.mp4"
    video_path = os.path.join(VIDEO_SAVE_DIRECTORY, video_filename)

    # Check if video already exists
    if os.path.exists(video_path):
        print("Video already exists in the directory.")
        return os.path.relpath(video_path)  # Return relative path to existing video

    # Get the highest resolution stream of the YouTube video
    video = YouTube(video_url)
    video = video.streams.get_highest_resolution()

    try:
        # Download the YouTube video to the specified directory
        video_path = video.download(VIDEO_SAVE_DIRECTORY)
        print(f"Video downloaded successfully as {video_path}.")
        return os.path.relpath(video_path)  # Return relative path to newly saved video
    except Exception as e:
        print(f"Failed to download video due to {e}")
        return None  # Return None if download fails
    

from_rapid_api_Video_download_link_from_instagaram('https://www.instagram.com/p/C39tHnbsjOB')