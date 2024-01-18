import os
import glob
import ntpath
import shutil
import subprocess
from pathlib import Path

import easygui
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import detector


# ===== constants
TEMP_PROCESSED_VIDEO = str(Path('./temp/processed.mp4'))
TEMP_PROCESSED_H264 = str(Path('./temp/processed_h264.mp4'))

coord_directions = [
    'top-left',
    'top-right',
    'bottom-left',
    'bottom-right'
]

coord_dir_colors = [
    (255, 0, 0), # red
    (0, 255, 0), # green
    (0, 0, 255), # blue
    (0, 255, 255), # light_blue
]


# ===== create needed folder 
if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("results"):
    os.makedirs("results")

# ===== streamlit status management
# init 
if 'init' not in st.session_state:
    st.session_state.init = True
    
def initalize():
    # clear working dir
    files = glob.glob('temp/*')
    for f in files:
        os.remove(f)
    files = glob.glob('results/*')
    for f in files:
        shutil.rmtree(f)
        
if st.session_state.init:
    st.session_state.init = False
    initalize()

# buttons
if 'video_submit' not in st.session_state:
    st.session_state.video_submit = False
if 'para_submit' not in st.session_state:
    st.session_state.para_submit = False
if 'save_submit' not in st.session_state:
    st.session_state.save_submit = False    

# videos
if 'cur_frames' not in st.session_state:
    st.session_state.cur_frames = []
if 'file_path' not in st.session_state:
    st.session_state.file_path = ''
if 'vid_width' not in st.session_state:
    st.session_state.vid_width = 0
if 'vid_height' not in st.session_state:
    st.session_state.vid_height = 0
if 'process_status' not in st.session_state:
    st.session_state.process_status = None
if 'save_counts' not in st.session_state:
    st.session_state.save_counts = 0

def reset_vid_status():
    st.session_state.cur_frames = []
    st.session_state.file_path = ''
    st.session_state.vid_width = 0
    st.session_state.vid_height = 0
    st.session_state.process_status = None
    st.session_state.para_submit = False
    st.session_state.save_submit = False
    st.session_state.frame_coord = {}
    st.session_state.coord_counters = 0
    st.session_state.coord_button = False

# results
if 'res_coverRatio' not in st.session_state:
    st.session_state.res_coverRatio = []
if 'res_areas' not in st.session_state:
    st.session_state.res_areas = []
if 'res_saver' not in st.session_state:
    st.session_state.res_saver = dict()

def reset_results():
    st.session_state.res_coverRatio = []
    st.session_state.res_areas = []

# ===== main
st.set_page_config(layout="wide")
st.title('Dryout area detection')

# read files
if st.button('Select your video file'):
    reset_vid_status()
    file_path = easygui.fileopenbox(title='Select your file', default="*.avi")
    st.session_state.file_path = file_path
    
    try:
        width, height, frames = detector.read_video(file_path)
        st.session_state.vid_width = width
        st.session_state.vid_height = height
        st.session_state.cur_frames = frames
        st.session_state.video_submit = True
    except:
        st.session_state.cur_frames = []
        st.session_state.video_submit = False
        
# read files state inform
if st.session_state.video_submit:
    st.write(f'File you selected: {st.session_state.file_path}')
    st.write(f'File {st.session_state.file_path} read!')
else:
    st.write(f'File you selected: {st.session_state.file_path}')
    st.write("File not yet selected or invalid file selected!")

# video, control panels
vid_panel, ctrl_panel = st.columns([3, 1])
vid_panel.header("Results")
ctrl_panel.header("Parameters")

with vid_panel:
    coord_vid, vid_tab1, vid_tab2, vid_tab3 = st.tabs(['Select coordinate', 'Processed video', 'Plots', 'History'])

with coord_vid:
    if st.session_state.video_submit and len(st.session_state.cur_frames):
        first_f = st.session_state.cur_frames[0].copy()
        
        # display on the image
        print(st.session_state.frame_coord)
        for (direction, coords), color in zip(st.session_state.frame_coord.items(), coord_dir_colors):
            coord_x, coord_y = coords
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"  {direction}: {coord_x}, {coord_y}"
            # put coordinate
            cv2.putText(first_f, 
                        text, 
                        (coord_x, coord_y), 
                        font, 
                        0.75,  
                        color,
                        2
                    )
            # put dot
            first_f = cv2.circle(first_f,
                                 (coord_x, coord_y), 
                                 5,
                                 color,
                                 -1) 

        cur_coords = streamlit_image_coordinates(first_f, key = "numpy")
        
        if st.session_state.coord_button:
            st.session_state.coord_button = False
            cur_coords = None
        
        if cur_coords is not None and st.session_state.coord_counters < 4:
            cur_dir = coord_directions[st.session_state.coord_counters]
            points = [cur_coords['x'], cur_coords['y']]
            if points not in st.session_state.frame_coord.values():
                st.session_state.frame_coord[cur_dir] = points
                st.session_state.coord_counters += 1
                st.rerun()

        if st.button("Reset"):
            st.session_state.frame_coord = {}
            st.session_state.coord_counters = 0
            st.session_state.coord_button = True
            st.rerun()
            
        
with ctrl_panel:
    # parameters
    dryout_pixel_range_slider = st.slider("Pixel bounds of dryout areas", 0, 255, (30, 100))
    dryout_area_threshold_ratio = st.slider("Minimum dryout area ratio compare with the detected square", 20, 300, 250, 5)
    dryout_area_quantile = st.slider("Dryout pixel quantile ", 10, 100, 40, 5)
    left_darken_gradient_mask_rate = st.slider("Rate of darkened gradient mask from the left", 0.0, 0.3, 0.03, 0.01)
    top_brighten_gradient_mask_rate = st.slider("Rate of brighten gradient mask from the top", 0.0, 0.3, 0.0, 0.01)
    bottom_brighten_gradient_mask_rate = st.slider("Rate of brighten gradient mask from the bottom", 0.0, 0.3, 0.0, 0.01)
    
    # process the video
    if st.button("Submit"):
        
        # check if the 4 coordinates are selected
        if len(st.session_state.frame_coord) != 4:
            st.write('Missing coordinate! Make sure you selected all 4 coordinates')
            
        else:
            reset_results()
            st.session_state.para_submit = True
            st.session_state.process_status = st.status('Processing dryout detection...', expanded = True) 
            with st.session_state.process_status as status:
            
                # clear temp folder
                files = glob.glob('temp/*')
                for f in files:
                    os.remove(f)
                
                st.write('Processing vid...')
                if len(st.session_state.cur_frames):
                    st.session_state.save_counts += 1
                    progress_bar = st.progress(0.0)
                    processed_frames = []
                    processed_frames_id = []
                    for fi, frame in enumerate(st.session_state.cur_frames):
                        covered_ratio, area, processed_frame = detector.dryout_detection(frame,
                                                                                        st.session_state.frame_coord,
                                                                                        dryout_pixel_range_slider[0],
                                                                                        dryout_pixel_range_slider[1],
                                                                                        dryout_area_threshold_ratio,
                                                                                        dryout_area_quantile,
                                                                                        left_darken_gradient_mask_rate,
                                                                                        top_brighten_gradient_mask_rate,
                                                                                        bottom_brighten_gradient_mask_rate)
                        
                        if covered_ratio == -1:
                            break
                        # keep all the frame
                        # elif covered_ratio == 0:
                        #     continue
                        else:
                            st.session_state.res_coverRatio.append(covered_ratio)
                            processed_frames_id.append(fi)
                            processed_frames.append(processed_frame)
                            st.session_state.res_areas.append(area)
                            
                        progress_bar.progress(fi / len(st.session_state.cur_frames),
                                            text = f"{fi + 1} of {len(st.session_state.cur_frames)} processed...")
                    progress_bar.progress(1.0,
                                        text = f"{len(st.session_state.cur_frames)} of {len(st.session_state.cur_frames)} processed...")   
                    
                    # save the processed vid as MP4
                    st.write('Saving the processed video as MP4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_vid = cv2.VideoWriter(TEMP_PROCESSED_VIDEO, 
                                            fourcc, 
                                            20.0, 
                                            (st.session_state.vid_width * 2, st.session_state.vid_height * 2))
                    
                    
                    if len(processed_frames) == 0:
                        status.update(label = "Bad parameters, nothing detected!", state = 'error', expanded = False)
                        fig, ax1 = plt.subplots()
                        st.session_state.res_saver[st.session_state.save_counts] = {
                            'status': 'failed',
                            'reason': 'Bad parameters',
                            'filename': os.path.splitext(ntpath.basename(st.session_state.file_path))[0],
                            'fig': fig,
                            'configs' : pd.DataFrame(
                                {
                                    'Paras': [
                                            'Pixel UPPER bound of dryout areas',
                                            'Pixel LOWER bound of dryout areas',
                                            'Minimum dryout area ratio to the selected square',
                                            'Dryout pixel quantile',
                                            'Rate of darkended gradient mask (from left)',
                                            'Rate of brightened gradient mask (from top)',
                                            'Rate of brightened gradient mask (from bottom)'
                                            ],
                                    "Values": [
                                            dryout_pixel_range_slider[0],
                                            dryout_pixel_range_slider[1],
                                            dryout_area_threshold_ratio,
                                            dryout_area_quantile,
                                            left_darken_gradient_mask_rate,
                                            top_brighten_gradient_mask_rate,
                                            bottom_brighten_gradient_mask_rate
                                            ]
                                }
                            )
                        }
                        
                        
                    else:
                        for f in processed_frames:
                            if f is not None:
                                out_vid.write(f)
                            else:
                                break
                        # Re-encode the mp4 to H264 for streamlit demo
                        st.write('Re-encoded to H264 for streamlit display')
                        out_vid.release()
                        subprocess.call(args=f"ffmpeg -y -i {TEMP_PROCESSED_VIDEO} -c:v libx264 {TEMP_PROCESSED_H264}".split(" "))
                        vid_tab1.video(TEMP_PROCESSED_H264)
                        
                        # create the cover ratio plot
                        st.write('Creating the plot for coverage and dryout areas')
                        fig, ax1 = plt.subplots()
                        ax1.title.set_text(ntpath.basename(st.session_state.file_path))
                        color = 'tab:blue'
                        ax1.set_xlabel("Frame")
                        ax1.set_ylabel("Coverage", color = color)
                        ax1.plot(range(1, len(st.session_state.res_coverRatio)+1), st.session_state.res_coverRatio, color = color) # left y axis for coverage
                        ax1.set_ylim([0, 1])
                        
                        color = 'tab:orange'
                        ax2 = ax1.twinx() # right y axis for areas
                        ax2.set_ylabel('Areas', color = color)
                        ax2.bar(range(1, len(st.session_state.res_areas)+1), st.session_state.res_areas, color = color, alpha = 0.5)
                        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                        
                        fig.tight_layout()
                        vid_tab2.pyplot(fig)
                        
                        # Saving result
                        filename = os.path.splitext(ntpath.basename(st.session_state.file_path))[0]
                        vid_name = f'processed_{ntpath.basename(st.session_state.file_path)}'
                        folder_name = f'trial{st.session_state.save_counts}_{filename}'
                        rec_name = f"records_{filename}.csv"
                        
                        if not os.path.exists(f'./results/{folder_name}'):
                            os.makedirs(f"./results/{folder_name}")
                        
                        shutil.copy(TEMP_PROCESSED_VIDEO, f'./results/{folder_name}/{vid_name}')
                        out = pd.DataFrame(
                            {
                                'frame': processed_frames_id,
                                'ratio': st.session_state.res_coverRatio,
                                'areas': st.session_state.res_areas,
                            }
                        )
                        
                        configs = pd.DataFrame(
                                {
                                    'Paras': [
                                            'Pixel UPPER bound of dryout areas',
                                            'Pixel LOWER bound of dryout areas',
                                            'Minimum dryout area ratio to the selected square',
                                            'Dryout pixel quantile',
                                            'Rate of darkended gradient mask (from left)',
                                            'Rate of brightened gradient mask (from top)',
                                            'Rate of brightened gradient mask (from bottom)'
                                            ],
                                    "Values": [
                                            dryout_pixel_range_slider[0],
                                            dryout_pixel_range_slider[1],
                                            dryout_area_threshold_ratio,
                                            dryout_area_quantile,
                                            left_darken_gradient_mask_rate,
                                            top_brighten_gradient_mask_rate,
                                            bottom_brighten_gradient_mask_rate
                                            ]
                                }
                            )
                        
                        out.to_csv(f"./results/{folder_name}/{rec_name}", index = False)
                        fig.savefig(f'./results/{folder_name}/{filename}.png')
                        configs.to_csv(f"./results/{folder_name}/configs.csv", index = False)
                        
                        st.write('Saving results')
                        st.session_state.res_saver[st.session_state.save_counts] = {
                            'status': 'success',
                            'filename': os.path.splitext(ntpath.basename(st.session_state.file_path))[0],
                            'fig': fig,
                            'configs': configs,
                        }

                        status.update(label = "All done!", state = 'complete', expanded = True)
                        
                else:
                    status.update(label = "Something went wrong, please check your file!", state = 'error', expanded = False)


# display history results
if st.session_state.save_counts:
    with vid_tab3:
        for i in range(1, st.session_state.save_counts + 1):
            filename = st.session_state.res_saver[i]["filename"]
            status = st.session_state.res_saver[i]['status']
            
            if status == 'success':
                with st.expander(label = f'Status: Success - trial: {i} / file: {filename}'):
                    plot_panel, config_panel = st.columns([3,1])
                    plot_panel.pyplot(st.session_state.res_saver[i]['fig'])
                    config_panel.dataframe(st.session_state.res_saver[i]['configs'])
            else:
                with st.expander(label = f'Status: Failed - trial: {i} / file: {filename}'):
                    reason = st.session_state.res_saver[i]['reason']
                    st.write(f'Reason: {reason}')
                    plot_panel, config_panel = st.columns([3,1])
                    plot_panel.pyplot(st.session_state.res_saver[i]['fig'])
                    config_panel.dataframe(st.session_state.res_saver[i]['configs'])