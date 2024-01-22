import gradio as gr
import torch
import imageio
#except:
  #!pip install imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
#from IPython.display import HTML
import warnings

with gr.Blocks() as demo:
    with gr.Row():
        refImg0 = gr.Image(type="pil")
        refImg1 = gr.Image(type="pil")
        refImg2 = gr.Image(type="pil")
        refImg3 = gr.Image(type="pil")
    
    with gr.Row():
        refVideo = gr.Video()
        img_input = gr.Image(type="pil")
    
    with gr.Row():
        find_best_frame = gr.Checkbox(False,label="better quality")
        createVideo = gr.Checkbox(True,label="create video")

    video_btn = gr.Button("Create from video")
    image_btn = gr.Button("create from image")
    selected_section = gr.Textbox(label="Selected Section")

    # def make_image(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode = 'relative'):
    #     assert mode in ['standard', 'relative', 'avd']
    #     with torch.no_grad():
    #         predictions = []
    #         source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    #         source = source.to(device)
    #         driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
    #         kp_source = kp_detector(source)
    #         kp_driving_initial = kp_detector(driving[:, :, 0])

    #         for frame_idx in tqdm(range(driving.shape[2])):
    #             driving_frame = driving[:, :, frame_idx]
    #             driving_frame = driving_frame.to(device)
    #             kp_driving = kp_detector(driving_frame)
    #             if mode == 'standard':
    #                 kp_norm = kp_driving
    #             elif mode=='relative':
    #                 kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
    #                                     kp_driving_initial=kp_driving_initial)
    #             elif mode == 'avd':
    #                 kp_norm = avd_network(kp_source, kp_driving)
    #             dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
    #                                                     kp_source=kp_source, bg_param = None, 
    #                                                     dropout_flag = False)
    #             out = inpainting_network(source, dense_motion)

    #             predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    #     return predictions



    def transImage(img_ref0,img_ref1,img_ref2,img_ref3,img_input):
        device = torch.device('cuda:0')
        dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
        #source_image_path = './assets/source5.png'
        #driving_video_path = './assets/driving.mp4'
        #output_video_path = './generated.mp4'
        output_image_path = './images/'
        config_path = 'config/vox-256.yaml'
        checkpoint_path = 'checkpoints/vox.pth.tar'
        predict_mode = 'relative'
        #find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result
        pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
        if(dataset_name == 'ted'): # for ted, the resolution is 384*384
            pixel = 384

        warnings.filterwarnings("ignore")

        source_image = np.asarray_chkfinite(img_input,dtype='uint8')
        ref_image0 = np.asarray_chkfinite(img_ref0,dtype='uint8')
        ref_image1 = np.asarray_chkfinite(img_ref1,dtype='uint8')
        ref_image2 = np.asarray_chkfinite(img_ref2,dtype='uint8')
        ref_image3 = np.asarray_chkfinite(img_ref3,dtype='uint8')

        source_image = resize(source_image, (pixel, pixel))[..., :3]
        ref_image0 = resize(ref_image0, (pixel, pixel))[..., :3]
        ref_image1 = resize(ref_image1, (pixel, pixel))[..., :3]
        ref_image2 = resize(ref_image2, (pixel, pixel))[..., :3]
        ref_image3 = resize(ref_image3, (pixel, pixel))[..., :3]
        


        driving_video = []

        driving_video = [ref_image0,ref_image1,ref_image2,ref_image3]

        from demo import load_checkpoints
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

        from demo import make_animation
        from skimage import img_as_ubyte

        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

        for i in range(len(predictions)):
            imageio.imsave(output_image_path + str(i) +'.png',img_as_ubyte(predictions[i]))

        return "ok"


    def transAnimation(refVideo,img_input,find_best_frame,createVideo):
        device = torch.device('cuda:0')
        dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
        #source_image_path = './assets/source5.png'
        #driving_video_path = './assets/driving.mp4'
        output_video_path = './generated.mp4'
        output_image_path = './frames/'
        config_path = 'config/vox-256.yaml'
        checkpoint_path = 'checkpoints/vox.pth.tar'
        predict_mode = 'relative' # ['standard', 'relative', 'avd']
        #find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result
        pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
        if(dataset_name == 'ted'): # for ted, the resolution is 384*384
            pixel = 384

        if find_best_frame:
            print("ok")

        warnings.filterwarnings("ignore")

        source_image = np.asarray_chkfinite(img_input,dtype='uint8')
        #source_image = imageio.imread('./assets/source5.png')
        reader = imageio.get_reader(refVideo)

        source_image = resize(source_image, (pixel, pixel))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

        from demo import load_checkpoints
        inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

        from demo import make_animation
        from skimage import img_as_ubyte

        if predict_mode=='relative' and find_best_frame:
            from demo import find_best_frame as _find
            i = _find(source_image, driving_video, device.type=='cpu')
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

        for i in range(len(predictions)):
            imageio.imsave(output_image_path + str(i) +'.png',img_as_ubyte(predictions[i]))
        imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

        return "ok"


       
    video_btn.click(transAnimation,[refVideo,img_input,find_best_frame,createVideo],selected_section)
    image_btn.click(transImage,[refImg0,refImg1,refImg2,refImg3,img_input],selected_section)



    def display(source, driving, generated=None):
        fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

        ims = []
        for i in range(len(driving)):
            cols = [source]
            cols.append(driving[i])
            if generated is not None:
                cols.append(generated[i])
            im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
            plt.axis('off')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
        plt.close()
        return ani


demo.launch()
#save resulting video
#print(len(predictions[1]))

  