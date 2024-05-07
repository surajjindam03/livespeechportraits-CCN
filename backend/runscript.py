import torch
import coremltools as ct
from options.test_audio2feature_options import TestOptions as FeatureOptions

# Define the path for the pre-trained model weights
model_weights_path = r"C:\Users\suraj\Documents\Suraj\Spring-2024-sem\CCN\Project\LiveSpeechPortraits-main\data\May\checkpoints\Audio2Feature\500_Audio2Feature.pkl"

# Parse the options
Featopt = FeatureOptions().parse()

# Load the entire model directly from the .pkl file using torch.load()
loaded_model_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

# Create a generic torch.nn.Module instance
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here
        # Example:
        # self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Define the forward pass of your model
        # Example:
        # x = self.conv1(x)
        return x

# Create the model instance
Audio2Feature = MyModel()

# Load weights into the model
with torch.no_grad():
    for name, param in Audio2Feature.named_parameters():
        param.copy_(loaded_model_state_dict[name])

# Set the model to evaluation mode
Audio2Feature.eval()

# Create an example input tensor (adjust the shape based on your input)
example_input = torch.randn(1, 512, 80)

# Trace the model
traced_model = torch.jit.trace(Audio2Feature, example_input)

# Convert using the lower-level CoreML Tools API
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    # minimum_deployment_target="ios13"  # Uncomment and set your target version if needed
)

# Define the path for saving the Core ML model
coreml_model_path = r"C:\Users\suraj\Documents\Suraj\Spring-2024-sem\CCN\Project\LiveSpeechPortraits-main\data\May\Audio2Feature.mlmodel"

# Save the Core ML model
coreml_model.save(coreml_model_path)

print("Core ML model conversion successful!")