import cv2
import numpy as np
from create_collapse_pyramids import *
from filters import *
from amplification import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='magnify color or small movement')
    parser.add_argument('input', type=str, help='input mp4 video')
    args = parser.parse_args()
    video_file = args.input
    output_filename = args.input.split('.')[0] + '_output.mp4'
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'fps of video: {fps}')

    #### Config params ####
    levels = 3
    low_f = 1.3
    high_f = 2
    amplification = [12,0,0]
    chrome_attenuation = .8
    ########################
    frame_cntr = 0

    pyd_dict = {i:[] for i in range(levels)}
    filt_pyd_dict = {i:[] for i in range(levels)}

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (540, 960))
        lp_pyd = create_laplacian_pyd(frame, levels)

        for i in range(levels):
            pyd_dict[i].append(lp_pyd[i])

        frame_cntr+=1

    for i in range(levels):
        print(f'processing level: {i}')
        pyd_dict[i] = np.array(pyd_dict[i])
        if amplification[i] > 0 and i==0:
            filt_pyd_dict[i] = ideal_temporal_filter(pyd_dict[i], fps, low_f, high_f)
            filt_pyd_dict[i] = color_amplification(filt_pyd_dict[i], amplification[i], chrome_attenuation)
            filt_pyd_dict[i] = pyd_dict[i] + filt_pyd_dict[i]
        else:
            filt_pyd_dict[i] = pyd_dict[i]

    filt_frame_list = []

    for i in range(pyd_dict[0].shape[0]):
        pyd = []
        for j in range(levels):
            pyd.append(filt_pyd_dict[j][i])
        filt_img = collapse_laplacian_pyd(pyd, levels)
        filt_frame_list.append(filt_img)

    cap.release()
    print(f'number of frames processed: {frame_cntr}')
    print('saving video')

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    h,w = filt_frame_list[0].shape[:2]
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (w,h), 1)
    for i in range(len(filt_frame_list)):
        writer.write(cv2.convertScaleAbs(filt_frame_list[i]))
    writer.release()

