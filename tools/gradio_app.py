import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_first_frame(video_path):
    """Extract and return the first frame of the video"""
    if not video_path:
        return None
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
        
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

class VOS:
    def __init__(self, sam2_cfg, sam2_checkpoint, device="cuda"):
        self.sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        
        hydra_overrides_extra = ["++model.non_overlap_masks=false"]
        self.video_predictor = build_sam2_video_predictor(
            config_file=sam2_cfg,
            ckpt_path=sam2_checkpoint,
            apply_postprocessing=True,
            hydra_overrides_extra=hydra_overrides_extra,
            device=device
        )

    def process_video(self, video_path, points, num_pathway=3, iou_thre=0.1, uncertainty=2):
        temp_dir = os.path.join(os.path.dirname(video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video")
            
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        height, width = first_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.image_predictor.set_image(first_frame_rgb)
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        masks, scores, logits = self.image_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        best_mask = masks[scores.argmax()]
        
        frame_count = 0
        current_frame = first_frame
        
        frame_name = f"{frame_count:05d}.jpg"
        frame_path = os.path.join(temp_dir, frame_name)
        cv2.imwrite(frame_path, current_frame)
        frame_count += 1
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            frame_name = f"{frame_count:05d}.jpg"
            frame_path = os.path.join(temp_dir, frame_name)
            cv2.imwrite(frame_path, current_frame)
            frame_count += 1
            
        cap.release()

        inference_state = self.video_predictor.init_state(
            video_path=temp_dir, 
            async_loading_frames=False
        )
        
        inference_state['num_pathway'] = num_pathway
        inference_state['iou_thre'] = iou_thre
        inference_state['uncertainty'] = uncertainty
        
        self.video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=best_mask
        )
        
        all_masks = []
        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0).cpu().numpy()
            all_masks.append(mask)
            
        import shutil
        shutil.rmtree(temp_dir)
            
        return all_masks, fps, (height, width)

def process_click(image, evt: gr.SelectData):
    """Process click events on the image"""
    points = getattr(process_click, 'points', [])
    points.append(evt.index)
    
    # Create visualization of points
    img = np.copy(image)
    for point in points:
        cv2.circle(img, point, 5, (0, 255, 0), -1)
    
    return img, points

def clear_points(image):
    """Clear all points from the image"""
    process_click.points = []
    return image, []

def process_video(video_path, points, sam2_cfg, sam2_checkpoint):
    """Process video with the given points"""
    if not points:
        return None, "Please select at least one point on the first frame"
    
    vos = VOS(sam2_cfg, sam2_checkpoint, device=DEVICE)
    points = np.array(points)
    
    try:
        masks, fps, (height, width) = vos.process_video(video_path, points)
        
        # Create output video with mask overlay
        cap = cv2.VideoCapture(video_path)
        output_path = "output_masked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for mask in masks:
            ret, frame = cap.read()
            if not ret:
                break
                
            mask_vis = np.zeros_like(frame)
            mask_vis[:,:,1] = mask.astype(np.uint8) * 255
            frame_with_mask = cv2.addWeighted(frame, 1, mask_vis, 0.5, 0)
            out.write(frame_with_mask)
        
        cap.release()
        out.release()
        
        return output_path, "Processing complete"
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def create_interface():
    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center;'>Video Object Segmentation with SAM2</h1>")
        gr.Image("img/logo.png", show_label=False, width=50)
        gr.Markdown("1. Upload a video\n2. Click points on the object you want to track\n3. Click Process to generate the masked video")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                first_frame = gr.Image(label="Select Points on First Frame", interactive=True, type="numpy")
                points_state = gr.State([])
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Points")
                    process_btn = gr.Button("Process Video", variant="primary")
                
            with gr.Column():
                video_output = gr.Video(label="Output Video")
                status = gr.Textbox(label="Status")
        
        # When video is uploaded, extract and show first frame
        video_input.change(
            fn=get_first_frame,
            inputs=[video_input],
            outputs=[first_frame]
        )
        
        # Handle point selection
        first_frame.select(
            process_click,
            inputs=[first_frame],
            outputs=[first_frame, points_state]
        )
        
        # Clear points
        clear_btn.click(
            clear_points,
            inputs=[first_frame],
            outputs=[first_frame, points_state]
        )
        
        # Process video
        process_btn.click(
            process_video,
            inputs=[
                video_input,
                points_state,
                gr.Textbox(value="configs/sam2.1/sam2.1_hiera_b+.yaml", visible=False),
                gr.Textbox(value="./checkpoints/sam2.1_hiera_base_plus.pt", visible=False)
            ],
            outputs=[video_output, status]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()