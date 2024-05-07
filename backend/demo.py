import os  # Module for interacting with the operating system
import subprocess  # Module for running external commands
from os.path import join  # Function to join path components
from tqdm import tqdm  # Module for progress bars
import numpy as np  # Module for numerical operations
import torch  # Module for PyTorch
from collections import OrderedDict  # Module for ordered dictionaries
import librosa  # Module for audio processing
from skimage.io import imread  # Function to read images
import cv2  # Module for computer vision tasks
import scipy.io as sio  # Module for file I/O
import argparse  # Module for command-line argument parsing
import yaml  # Module for parsing YAML files
import albumentations as A  # Module for image augmentations
import albumentations.pytorch  # Module for PyTorch-compatible image augmentations
from pathlib import Path  # Module for object-oriented path operations

from options.test_audio2feature_options import TestOptions as FeatureOptions  # Import TestOptions as FeatureOptions
from options.test_audio2headpose_options import TestOptions as HeadposeOptions  # Import TestOptions as HeadposeOptions
from options.test_feature2face_options import TestOptions as RenderOptions  # Import TestOptions as RenderOptions

from datasets import create_dataset  # Import function to create dataset
from models import create_model  # Import function to create model
from models.networks import APC_encoder  # Import APC_encoder model
import util.util as util  # Import utility functions
from util.visualizer import Visualizer  # Import Visualizer class
from funcs import utils  # Import utility functions
from funcs import audio_funcs  # Import audio-related functions

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

def write_video_with_audio(audio_path, output_path, prefix='pred_'):
    fps, fourcc = 60, cv2.VideoWriter_fourcc(*'DIVX')  # Set video properties
    video_tmp_path = join(save_root, 'tmp.avi')  # Temporary video file path
    out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (Renderopt.loadSize, Renderopt.loadSize))  # Create video writer object
    for j in tqdm(range(nframe), position=0, desc='writing video'):  # Loop over frames
        img = cv2.imread(join(save_root, prefix + str(j+1) + '.jpg'))  # Read frame image
        out.write(img)  # Write frame to video
    out.release()  # Release video writer
    cmd = 'ffmpeg -i "' + video_tmp_path + '" -i "' + audio_path + '" -codec copy -shortest "' + output_path + '"'  # Command to merge video and audio
    subprocess.call(cmd, shell=True)  # Execute command
    os.remove(video_tmp_path)  # Remove temporary video file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # Create argument parser
    parser.add_argument('--id', default='May', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")  # Add argument for person ID
    parser.add_argument('--driving_audio', default='./data/input/00083.wav', help="path to driving audio")  # Add argument for driving audio path
    parser.add_argument('--save_intermediates', default=0, help="whether to save intermediate results")  # Add argument for saving intermediate results
    parser.add_argument('--device', type=str, default='cpu', help='use cuda for GPU or use cpu for CPU')  # Add argument for device (CPU or GPU)

    ############################### I/O Settings ##############################
    # load config files
    opt = parser.parse_args()  # Parse command-line arguments
    device = torch.device(opt.device)  # Set device (CPU or GPU)
    with open(join('./config/', opt.id + '.yaml')) as f:
        config = yaml.safe_load(f)  # Load configuration file
    data_root = join('./data/', opt.id)  # Set data root directory
    # create the results folder
    audio_name = os.path.split(opt.driving_audio)[1][:-4]  # Get audio name
    save_root = join('./results/', opt.id, audio_name)  # Set results directory
    if not os.path.exists(save_root):
        os.makedirs(save_root)  # Create results directory if it doesn't exist

    ############################ Hyper Parameters #############################
    h, w, sr, FPS = 512, 512, 16000, 60  # Set image size, sampling rate, and frames per second
    mouth_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])  # Define mouth indices
    eye_brow_indices = [27, 65, 28, 68, 29, 67, 30, 66, 31, 72, 32, 69, 33, 70, 34, 71]  # Define eyebrow indices
    eye_brow_indices = np.array(eye_brow_indices, np.int32)  # Convert eyebrow indices to numpy array

    ############################ Pre-defined Data #############################
    mean_pts3d = np.load(join(data_root, 'mean_pts3d.npy'))  # Load mean 3D points
    fit_data = np.load(config['dataset_params']['fit_data_path'])  # Load fit data
    pts3d = np.load(config['dataset_params']['pts3d_path']) - mean_pts3d  # Load 3D points and subtract mean
    trans = fit_data['trans'][:,:,0].astype(np.float32)  # Load translation parameters
    mean_translation = trans.mean(axis=0)  # Calculate mean translation
    candidate_eye_brow = pts3d[10:, eye_brow_indices]  # Extract candidate eyebrow points
    std_mean_pts3d = np.load(config['dataset_params']['pts3d_path']).mean(axis=0)  # Load mean 3D points
    # candidates images
    img_candidates = []  # Initialize list for candidate images
    for j in range(4):
        output = imread(join(data_root, 'candidates', f'normalized_full_{j}.jpg'))  # Read candidate image
        output = A.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 
                                                          'std':(0.5,0.5,0.5)})(image=output)['image']  # Apply tensor transformation
        img_candidates.append(output)  # Append transformed image to list
    img_candidates = torch.cat(img_candidates).unsqueeze(0).to(device)  # Convert list to tensor and move to device

    # shoulders
    shoulders = np.load(join(data_root, 'normalized_shoulder_points.npy'))  # Load normalized shoulder points
    shoulder3D = np.load(join(data_root, 'shoulder_points3D.npy'))[1]  # Load 3D shoulder points
    ref_trans = trans[1]  # Get reference translation

    # camera matrix, we always use training set intrinsic parameters.
    camera = utils.camera()  # Initialize camera object
    camera_intrinsic = np.load(join(data_root, 'camera_intrinsic.npy')).astype(np.float32)  # Load camera intrinsic parameters
    APC_feat_database = np.load(join(data_root, 'APC_feature_base.npy'))  # Load APC feature database

    # load reconstruction data
    scale = sio.loadmat(join(data_root, 'id_scale.mat'))['scale'][0,0]  # Load scale parameter
    # Audio2Mel_torch = audio_funcs.Audio2Mel(n_fft=512, hop_length=int(16000/120), win_length=int(16000/60), sampling_rate=16000, 
    #                                         n_mel_channels=80, mel_fmin=90, mel_fmax=7600.0).to(device)
	


    ########################### Experiment Settings ###########################
    #### user config
    # Load whether Locally Linear Embedding (LLE) should be used from the configuration
    use_LLE = config['model_params']['APC']['use_LLE']
    # Load the number of nearest neighbors to use in LLE
    Knear = config['model_params']['APC']['Knear']
    # Load the percentage of LLE to be applied from the configuration
    LLE_percent = config['model_params']['APC']['LLE_percent']
    # Load the standard deviation for headpose data smoothing
    headpose_sigma = config['model_params']['Headpose']['sigma']
    # Load the smoothing sigma value for audio to mouth feature smoothing
    Feat_smooth_sigma = config['model_params']['Audio2Mouth']['smooth']
    # Load the smoothing sigma value for headpose data
    Head_smooth_sigma = config['model_params']['Headpose']['smooth']
    # Initialize smoothing sigma values for feature center and head center to zero
    Feat_center_smooth_sigma, Head_center_smooth_sigma = 0, 0
    # Load the method for amplitude modulation from configuration for audio to mouth features
    AMP_method = config['model_params']['Audio2Mouth']['AMP'][0]
    # Load additional amplitude modulation settings for features
    Feat_AMPs = config['model_params']['Audio2Mouth']['AMP'][1:]
    # Load the rotation and translation amplitude parameters for headpose adjustments
    rot_AMP, trans_AMP = config['model_params']['Headpose']['AMP']
    # Load amplitude modulation for shoulder movement
    shoulder_AMP = config['model_params']['Headpose']['shoulder_AMP']
    # Load setting to determine if feature maps should be saved during Image-to-Image processing
    save_feature_maps = config['model_params']['Image2Image']['save_input']
    
    #### common settings
    # Parse feature options from predefined settings or defaults
    Featopt = FeatureOptions().parse() 
    # Parse headpose options from predefined settings or defaults
    Headopt = HeadposeOptions().parse()
    # Parse rendering options from predefined settings or defaults
    Renderopt = RenderOptions().parse()
    # Load the checkpoint path for audio to mouth features
    Featopt.load_epoch = config['model_params']['Audio2Mouth']['ckp_path']
    # Load the checkpoint path for headpose adjustments
    Headopt.load_epoch = config['model_params']['Headpose']['ckp_path']
    # Set the root directory for dataset used in rendering
    Renderopt.dataroot = config['dataset_params']['root']
    # Load the checkpoint path for image to image model
    Renderopt.load_epoch = config['model_params']['Image2Image']['ckp_path']
    # Set the image size for rendering
    Renderopt.size = config['model_params']['Image2Image']['size']
    ## GPU or CPU
    # Check if the configuration is set to use CPU, and if so, set all GPU IDs to empty lists to enforce CPU usage
    if opt.device == 'cpu':
        Featopt.gpu_ids = Headopt.gpu_ids = Renderopt.gpu_ids = []

    

    
    ############################# Load Models #################################
    # Print message indicating the loading of the APC model
    print('---------- Loading Model: APC-------------')
    # Instantiate the APC model with parameters from configuration
    APC_model = APC_encoder(config['model_params']['APC']['mel_dim'],
                            config['model_params']['APC']['hidden_size'],
                            config['model_params']['APC']['num_layers'],
                            config['model_params']['APC']['residual'])
    # Load the pretrained weights into the APC model, with leniency in matching
    APC_model.load_state_dict(torch.load(config['model_params']['APC']['ckp_path']), strict=False)
    # Move the model to GPU if CUDA is available
    if opt.device == 'cuda':
        APC_model.cuda() 
    # Set the model to evaluation mode
    APC_model.eval()
    # Print message indicating loading of the next model based on task
    print('---------- Loading Model: {} -------------'.format(Featopt.task))
    # Create the Audio2Feature model based on the Feature options
    Audio2Feature = create_model(Featopt)   
    # Setup the Audio2Feature model with the corresponding options
    Audio2Feature.setup(Featopt)  
    # Set the model to evaluation mode
    Audio2Feature.eval()     
    # Print message indicating loading of the Headpose model based on task
    print('---------- Loading Model: {} -------------'.format(Headopt.task))
    # Create the Audio2Headpose model based on the Headpose options
    Audio2Headpose = create_model(Headopt)    
    # Setup the Audio2Headpose model with the corresponding options
    Audio2Headpose.setup(Headopt)
    # Set the model to evaluation mode
    Audio2Headpose.eval()              
    # Special condition if the feature decoder is specified as WaveNet
    if Headopt.feature_decoder == 'WaveNet':
        # If using CUDA, retrieve the receptive field information from the CUDA module
        if opt.device == 'cuda':
            Headopt.A2H_receptive_field = Audio2Headpose.Audio2Headpose.module.WaveNet.receptive_field
        else:
            # Retrieve the receptive field information directly if not using CUDA
            Headopt.A2H_receptive_field = Audio2Headpose.Audio2Headpose.WaveNet.receptive_field
    # Print message indicating loading of the Rendering model based on task
    print('---------- Loading Model: {} -------------'.format(Renderopt.task))
    # Create a dataset for the face rendering based on Render options
    facedataset = create_dataset(Renderopt) 
    # Create the Feature2Face model based on the Render options
    Feature2Face = create_model(Renderopt)
    # Setup the Feature2Face model with the corresponding options
    Feature2Face.setup(Renderopt)   
    # Set the model to evaluation mode
    Feature2Face.eval()
    # Initialize a visualizer for rendering operations
    visualizer = Visualizer(Renderopt)

    
    
    
    ############################## Inference ##################################
    # Display message indicating the start of audio processing
    print('Processing audio: {} ...'.format(audio_name)) 
    # Load the audio file at the specified sample rate
    audio, _ = librosa.load(opt.driving_audio, sr=sr)
    # Calculate the total number of frames in the audio by converting duration to frames
    total_frames = np.int32(audio.shape[0] / sr * FPS) 
    
    
    #### 1. compute APC features   
    # Notify that APC features computation is starting
    print('1. Computing APC features...')                    
    # Compute Mel spectrogram from the audio
    mel80 = utils.compute_mel_one_sequence(audio, device=opt.device)
    # Get the number of frames in the Mel spectrogram
    mel_nframe = mel80.shape[0]
    with torch.no_grad():  # Disable gradient computation for inference
        length = torch.Tensor([mel_nframe])  # Create a tensor for sequence length
        mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to(device).unsqueeze(0)  # Convert Mel to tensor and add batch dimension
        hidden_reps = APC_model.forward(mel80_torch, length)[0]   # Obtain hidden representations from APC model
        hidden_reps = hidden_reps.cpu().numpy()  # Move the data to CPU and convert to numpy array
    audio_feats = hidden_reps  # Store audio features for further processing
            
    
    #### 2. manifold projection
    # Perform manifold projection if LLE is enabled
    if use_LLE:
        # Notify that manifold projection is starting
        print('2. Manifold projection...')
        ind = utils.KNN_with_torch(audio_feats, APC_feat_database, K=Knear)  # Find K-nearest neighbors
        weights, feat_fuse = utils.compute_LLE_projection_all_frame(audio_feats, APC_feat_database, ind, audio_feats.shape[0])  # Compute LLE projection
        audio_feats = audio_feats * (1-LLE_percent) + feat_fuse * LLE_percent  # Blend original and projected features
    
    
    #### 3. Audio2Mouth
    # Notify that Audio2Mouth inference is starting
    print('3. Audio2Mouth inference...')
    pred_Feat = Audio2Feature.generate_sequences(audio_feats, sr, FPS, fill_zero=True, opt=Featopt)  # Generate mouth shapes based on audio features
    
    
    #### 4. Audio2Headpose
    # Notify that Headpose inference is starting
    print('4. Headpose inference...')
    # Initialize previous headpose as zero (no prior head movement)
    pre_headpose = np.zeros(Headopt.A2H_wavenet_input_channels, np.float32)
    pred_Head = Audio2Headpose.generate_sequences(audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.3, opt=Headopt)  # Generate head poses based on audio features

       
    
    #### 5. Post-Processing 
    # Print to console the start of post-processing phase
    print('5. Post-processing...')
    # Determine the number of frames based on the minimum length of feature and head pose data
    nframe = min(pred_Feat.shape[0], pred_Head.shape[0])
    # Initialize an array to store 3D points for all frames and 73 landmarks
    pred_pts3d = np.zeros([nframe, 73, 3])
    # Fill the mouth part of the 3D points array with reshaped predicted features
    pred_pts3d[:, mouth_indices] = pred_Feat.reshape(-1, 25, 3)[:nframe]
    
    ## mouth
    # Apply smoothing to mouth landmarks in 3D points
    pred_pts3d = utils.landmark_smooth_3d(pred_pts3d, Feat_smooth_sigma, area='only_mouth')
    # Amplify mouth points motion based on the specified method and amplitude parameters
    pred_pts3d = utils.mouth_pts_AMP(pred_pts3d, True, AMP_method, Feat_AMPs)
    # Adjust the mouth positions by adding the mean 3D points
    pred_pts3d = pred_pts3d + mean_pts3d
    # Solve intersections in mouth points if they exist
    pred_pts3d = utils.solve_intersect_mouth(pred_pts3d)

    ## headpose
    # Amplify the rotation values of the head poses
    pred_Head[:, 0:3] *= rot_AMP
    # Amplify the translation values of the head poses
    pred_Head[:, 3:6] *= trans_AMP
    # Smooth the head pose data
    pred_headpose = utils.headpose_smooth(pred_Head[:,:6], Head_smooth_sigma).astype(np.float32)
    # Offset the translations by adding mean translation values
    pred_headpose[:, 3:] += mean_translation
    # Add 180 degrees to the first rotation component (usually yaw)
    pred_headpose[:, 0] += 180
    
    ## compute projected landmarks
    # Initialize an array for storing the projected 2D landmarks
    pred_landmarks = np.zeros([nframe, 73, 2], dtype=np.float32)
    # Initialize an array for adjusted 3D points, starting with standard mean points
    final_pts3d = np.zeros([nframe, 73, 3], dtype=np.float32)
    final_pts3d[:] = std_mean_pts3d.copy()
    # Update specific regions (typically around the eyes) with the predicted 3D points
    final_pts3d[:, 46:64] = pred_pts3d[:nframe, 46:64]
    # Iterate over all frames to project 3D landmarks to 2D using the camera model
    for k in tqdm(range(nframe)):
        ind = k % candidate_eye_brow.shape[0]
        # Add variations to the eyebrow regions from a set of candidates
        final_pts3d[k, eye_brow_indices] = candidate_eye_brow[ind] + mean_pts3d[eye_brow_indices]
        # Project 3D points to 2D using the intrinsic camera parameters and head pose
        pred_landmarks[k], _, _ = utils.project_landmarks(camera_intrinsic, camera.relative_rotation, 
                                                          camera.relative_translation, scale, 
                                                          pred_headpose[k], final_pts3d[k]) 
    
    ## Upper Body Motion
    # Initialize arrays to store projected 2D shoulder positions and their 3D counterparts
    pred_shoulders = np.zeros([nframe, 18, 2], dtype=np.float32)
    pred_shoulders3D = np.zeros([nframe, 18, 3], dtype=np.float32)
    # Calculate shoulder positions for each frame based on head pose changes
    for k in range(nframe):
        # Calculate the translation difference from a reference position
        diff_trans = pred_headpose[k][3:] - ref_trans
        # Apply amplification and calculate new 3D shoulder positions
        pred_shoulders3D[k] = shoulder3D + diff_trans * shoulder_AMP
        # Project 3D shoulder points to 2D
        project = camera_intrinsic.dot(pred_shoulders3D[k].T)
        project[:2, :] /= project[2, :]  # Normalize by Z to convert from homogeneous to Cartesian coordinates
        pred_shoulders[k] = project[:2, :].T

    

    #### 6. Image2Image translation & Saving results
# Notify the user of the current processing stage
print('6. Image2Image translation & Saving results...')
# Iterate over each frame to perform image-to-image translation and save the results
for ind in tqdm(range(0, nframe), desc='Image2Image translation inference'):
    # Retrieve the current frame's feature map from the dataset using predicted landmarks and shoulders
    current_pred_feature_map = facedataset.dataset.get_data_test_mode(pred_landmarks[ind], 
                                                                      pred_shoulders[ind], 
                                                                      facedataset.dataset.image_pad)
    # Prepare the feature map for input to the model by adding a batch dimension and sending to the device (GPU/CPU)
    input_feature_maps = current_pred_feature_map.unsqueeze(0).to(device)
    # Generate the predicted face from the feature map using the neural network
    pred_fake = Feature2Face.inference(input_feature_maps, img_candidates)
    # Prepare a list for storing images to visualize or save
    visual_list = [('pred', util.tensor2im(pred_fake[0]))]
    # Optionally save the input feature maps as images
    if save_feature_maps:
        visual_list += [('input', np.uint8(current_pred_feature_map[0].cpu().numpy() * 255))]
    # Create an ordered dictionary from the list to maintain the order
    visuals = OrderedDict(visual_list)
    # Save the images using the visualizer tool
    visualizer.save_images(save_root, visuals, str(ind+1))

## make videos
# Generate corresponding audio for the video, reusing the same clip for all results
tmp_audio_path = join(save_root, 'tmp.wav')
# Clip the audio to match the number of frames based on sample rate and FPS
tmp_audio_clip = audio[: np.int32(nframe * sr / FPS)]
# Write the clipped audio to a temporary file using soundfile library
import soundfile as sf
sf.write(tmp_audio_path, tmp_audio_clip, sr)

# Define the final path for the video file with audio
final_path = join(save_root, audio_name + '.avi')
# Create a video from the saved images and add audio
write_video_with_audio(tmp_audio_path, final_path, 'pred_')
# If saving feature maps, define the path and create the video for feature maps
feature_maps_path = join(save_root, audio_name + '_feature_maps.avi')
write_video_with_audio(tmp_audio_path, feature_maps_path, 'input_')

# Clean up the temporary audio file
if os.path.exists(tmp_audio_path):
    os.remove(tmp_audio_path)
# Optionally delete intermediate images if not needed for saving space
if not opt.save_intermediates:
    _img_paths = list(map(lambda x: str(x), list(Path(save_root).glob('*.jpg'))))
    for i in tqdm(range(len(_img_paths)), desc='deleting intermediate images'):
        os.remove(_img_paths[i])

# Indicate completion of the process
print('Finish!')

    


    
    



   


