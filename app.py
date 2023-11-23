import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Define your model class
class YourModelClass(torch.nn.Module):
    def __init__(self, num_classes):
        super(YourModelClass, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer for transfer learning
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Load the trained PyTorch model state dictionary
state_dict = torch.load("C:\\Users\\ashmi\\cv project\\model1.pth", map_location=torch.device('cuda'))

# Number of classes in your dataset
num_classes = 4  # Modify this according to your dataset

# Create an instance of your model class
model = YourModelClass(num_classes)

# Load the state dictionary, ensuring that keys match the model's structure
model.load_state_dict(state_dict, strict=False)  # Set strict=False to skip missing keys
model.eval()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a Streamlit app
st.title("Corn Plant Disease Classification")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output, 1).item()

    # Define class labels
    class_labels = {0: "Blight", 1: "Common Rust", 2: "Gray Leaf Spot", 3: "Healthy"}

    # Display the result
    st.write(f"Predicted Class: {class_labels[predicted_class]}")

# Add any additional content or information as needed
st.write("This Streamlit app classifies diseases in corn plants using a pre-trained ResNet-18 model for transfer learning.")

