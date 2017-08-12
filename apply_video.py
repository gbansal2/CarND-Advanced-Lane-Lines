from moviepy.editor import VideoFileClip
from lanelines import process_image


white_output = 'output_videos/project_video_lanes.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) 
white_clip.write_videofile(white_output, audio=False)
