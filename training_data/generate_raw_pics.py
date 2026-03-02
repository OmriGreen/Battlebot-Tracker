import numpy as np
import cv2 as cv
import gradio as gr
import os

if __name__ == '__main__':
    cv.namedWindow('image')
    fourcc = cv.VideoWriter_fourcc(*'MJPG')  

    # gets the names of all videos in the raw_video folder
    video_names = os.listdir('raw_video')
    video_names = [name for name in video_names if name.endswith('.mp4')]

    vid_counter = 0
    pic_counter = 0

    for video in video_names:
        pic_counter = 0

        frames = 0
        cap = cv.VideoCapture(f'raw_video/{video}')
        if not cap.isOpened():
            print(f"Cannot open video {video}")
            exit()


        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Can't receive frame (stream end?). Exiting ...")
                break

            # takes a screenshot and saves it every 20 seconds (assuming the video is 60 fps, that's every 1200 frames)
            if frames % 1200 == 0:
                cv.imwrite(f'raw_pics/video_{vid_counter}_{pic_counter}.jpg', frame)
                pic_counter += 1
            cv.imshow('image', frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            frames += 1
        print(f"Finished processing video {vid_counter}/{len(video_names)}: {video}")
        vid_counter += 1


    # cap = cv.VideoCapture('videos/BZ-nhrl_oct25_3lb-thepoweroffriendship-uhhh-4ca1-Cage-2-Overhead-High.mp4')
    # # cap = cv.VideoCapture('videos/BZ-nhrl_oct25_3lb-eruption-chrisgriffin2-W-40-Cage-2-Overhead-High.mp4')

    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    recording = 0