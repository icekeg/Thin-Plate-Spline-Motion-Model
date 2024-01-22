import gradio as gr
import torch
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
#from IPython.display import HTML
import warnings

with gr.Blocks() as demo:
    with gr.Row():
        refVideo = gr.Video()
        img_input = gr.Image(type="pil")
    
    with gr.Row():
        find_best_frame = gr.Checkbox(False,label="better quality")
        createVideo = gr.Checkbox(True,label="create video")

    start_btn = gr.Button("switch head")
    selected_section = gr.Textbox(label="Selected Section")

    def transAnimation(refVideo,img_input,find_best_frame,createVideo):
        source_image = imageio.imread('./assets/source.png')
        source_image2 = img_input
        print("Pic01 is:")
        print(type(source_image[0][0][0]))
        print(source_image)
        print("Pic02 is:")
        print(type(source_image2[0][0][0]))
        print(source_image2)
        pp2 = np.asarray_chkfinite(source_image2,dtype=int)
        print(pp2)
        
        
        
        #reader = imageio.get_reader(refVideo)

        source_image2 = resize(pp2, (256, 256))[..., :3]
        return "ok"
    
    start_btn.click(transAnimation,[refVideo,img_input,find_best_frame,createVideo],selected_section)

demo.launch()