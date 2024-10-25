import argparse
import os
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VOS:
    def __init__(self, sam2_cfg, sam2_checkpoint, device="cuda"):
        """Initialize both image and video predictors"""
        # Initialize image predictor for first frame
        self.sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Initialize video predictor for tracking
        hydra_overrides_extra = ["++model.non_overlap_masks=false"]
        self.video_predictor = build_sam2_video_predictor(
            config_file=sam2_cfg,
            ckpt_path=sam2_checkpoint,
            apply_postprocessing=True,
            hydra_overrides_extra=hydra_overrides_extra,
            device=device
        )

    def process_video(self, video_path, points, num_pathway=3, iou_thre=0.1, uncertainty=2):
        """
        Process a video with given initial points
        
        Args:
            video_path: Path to the video file
            points: List of [x,y] coordinates to track
            num_pathway: Number of segmentation pathways
            iou_thre: IoU threshold for filtering masks
            uncertainty: Uncertainty threshold for mask selection
        """
        # Create temporary directory for frames
        temp_dir = os.path.join(os.path.dirname(video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract first frame and get mask using points
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video")
            
        # Convert BGR to RGB
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        height, width = first_frame.shape[:2]
        
        # Get initial mask using SAM2 image predictor
        self.image_predictor.set_image(first_frame_rgb)
        input_points = np.array(points)
        input_labels = np.ones(len(points))  # Assuming all points are positive
        masks, scores, logits = self.image_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Get best mask
        best_mask = masks[scores.argmax()]
        
        # Save frames as images (needed for video predictor)
        frame_count = 0
        current_frame = first_frame  # Start with the first frame
        
        # Save the first frame
        frame_name = f"{frame_count:05d}.jpg"
        frame_path = os.path.join(temp_dir, frame_name)
        cv2.imwrite(frame_path, current_frame)
        frame_count += 1
        
        # Save the rest of the frames
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
                
            frame_name = f"{frame_count:05d}.jpg"
            frame_path = os.path.join(temp_dir, frame_name)
            cv2.imwrite(frame_path, current_frame)
            frame_count += 1
            
        cap.release()

        # Initialize video predictor state
        inference_state = self.video_predictor.init_state(
            video_path=temp_dir, 
            async_loading_frames=False
        )
        
        # Set parameters
        inference_state['num_pathway'] = num_pathway
        inference_state['iou_thre'] = iou_thre
        inference_state['uncertainty'] = uncertainty
        
        # Add initial mask
        self.video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,  # Single object tracking
            mask=best_mask
        )
        
        # Propagate through video
        all_masks = []
        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0).cpu().numpy()  # First mask only
            all_masks.append(mask)
            
        import shutil
        shutil.rmtree(temp_dir)
            
        return all_masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--points", type=float, nargs='+', required=True, 
                      help="List of x,y coordinates. Format: x1 y1 x2 y2 ...")
    parser.add_argument("--output_video", type=str, required=True, help="Path to output video")
    parser.add_argument("--sam2_cfg", type=str, 
                      default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                      help="SAM2 config path")
    parser.add_argument("--sam2_checkpoint", type=str,
                      default="./checkpoints/sam2.1_hiera_base_plus.pt",
                      help="SAM2 checkpoint path")
    parser.add_argument("--num_pathway", type=int, default=3)
    parser.add_argument("--iou_thre", type=float, default=0.1)
    parser.add_argument("--uncertainty", type=float, default=2)
    
    args = parser.parse_args()
    
    points = np.array(args.points).reshape(-1, 2)
    
    vos = VOS(args.sam2_cfg, args.sam2_checkpoint, device=DEVICE)
    
    # Process video
    masks = vos.process_video(
        args.video_path,
        points,
        num_pathway=args.num_pathway,
        iou_thre=args.iou_thre,
        uncertainty=args.uncertainty
    )
    
    # Create visualization video
    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
    
    # Initialize video writer with original FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # Process each frame
    for i, mask in enumerate(masks):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply mask overlay
        mask_vis = np.zeros_like(frame)
        mask_vis[:,:,1] = mask.astype(np.uint8) * 255  # Green channel
        frame_with_mask = cv2.addWeighted(frame, 1, mask_vis, 0.5, 0)
        
        out.write(frame_with_mask)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    main()