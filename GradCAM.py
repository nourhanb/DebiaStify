import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from data_handler.dataloader_factory import DataloaderFactory

# Data loader setup
tmp = DataloaderFactory.get_dataloader('celeba', img_size=176,
                                       batch_size=100, seed=0,
                                       num_workers=4,
                                       target='Wavy_Hair',
                                       skew_ratio=1,
                                       labelwise=True)

num_classes, num_groups, trainloader, val_loader = tmp

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model setup
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer for the number of classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Unpack the data based on your previous snippet
        inputs, _, attribute, labels, (index, image_name) = data
        inputs, labels, attribute = inputs.to(device), labels.to(device), attribute.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print("Finished Training")

# Save the model
model_path = 'resnet18_celeba.pth'
torch.save(model.state_dict(), model_path)


import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, HiResCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import allignments
from PIL import Image
import matplotlib.pyplot as plt
import torch
from networks.Resnet_alligned2 import resnet18 as custom_resnet18
# Load the model and set it to evaluation mode
#model = custom_resnet18(pretrained=False)
#model = allignments(model)
model=resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer for the number of classes
#model.load_state_dict(torch.load('/ubc/ece/home/ra/grads/nourhanb/Documents/distil/recent_teacher_student/trained_models/150524_Wavy_Hair/student_best_model_acc_0.81.pth'))
model.load_state_dict(torch.load('/ubc/ece/home/ra/grads/nourhanb/Documents/distil/Fairify/resnet18_celeba.pth'))
model.eval()


# Define the target layer for CAM (last layer of ResNet's layer4 block)
target_layers = [model.layer4[-1]]

# Define the path to the image you want to analyze
img_path = "/ubc/ece/home/ra/grads/nourhanb/Documents/distil/Fairify/data/wavy_hair.png"

# Load the image and convert it to RGB format
rgb_img = Image.open(img_path).convert('RGB')

# Define preprocessing transformations: resize, convert to tensor, normalize
preprocess = transforms.Compose([
    #transforms.Resize((218, 178)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply preprocessing transformations and add a batch dimension
input_tensor = preprocess(rgb_img).unsqueeze(0)

# Ensure the input tensor is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
model = model.to(device)

# Convert the PIL image to a NumPy array of type float32 and normalize to [0, 1]
rgb_img = np.array(rgb_img, dtype=np.float32) / 255.0

# Define the target class for which to generate the CAM
targets = [ClassifierOutputTarget(0)]

# List of CAM methods to be applied
cam_methods = {
    "GradCAM": GradCAM(model=model, target_layers=target_layers),
    "GradCAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
    "AblationCAM": AblationCAM(model=model, target_layers=target_layers),
    "HiResCAM": HiResCAM(model=model, target_layers=target_layers),
    "LayerCAM": LayerCAM(model=model, target_layers=target_layers),
}

# Create a figure to display the results
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Comparison of CAM Methods')

# Display the original RGB image
axs[0, 0].imshow(rgb_img)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')  # Hide the axis

# Apply each CAM method and visualize the output
for ax, (cam_name, cam_method) in zip(axs.flatten()[1:], cam_methods.items()):
    try:
        # Generate the CAM
        with cam_method:
            grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)

        # Extract the CAM for the single image in the batch
        grayscale_cam = grayscale_cam[0, :]

        # Overlay the grayscale CAM on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Display the visualization
        ax.imshow(visualization)
        ax.set_title(cam_name)
        ax.axis('off')  # Hide the axis
    except Exception as e:
        # In case of an error, print the error and skip this method
        print(f"Error with {cam_name}: {e}")
        ax.set_title(f"{cam_name} (error)")
        ax.axis('off')  # Hide the axis

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
output_path = 'cam_comparison_baseline.png'
plt.savefig(output_path)
plt.show()
 