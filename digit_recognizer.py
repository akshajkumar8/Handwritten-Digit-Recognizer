# Import required libraries
from tensorflow import keras  # For neural network model loading and inference
import tkinter as tk         # For GUI application framework
from PIL import Image, ImageDraw, ImageOps, ImageStat  # For image processing and manipulation
import numpy as np          # For numerical and array operations

# Load the MNIST-trained model for digit recognition
model = keras.models.load_model('mnist.h5')

def is_valid_digit(img):
    # Checks if the drawing could be a valid digit, filtering out empty drawings or random patterns
    if img.mode != 'L':
        img = img.convert('L')  # Convert to grayscale for analysis
    
    # Get image statistics to check content
    stats = ImageStat.Stat(img)
    
    # Return false if image is too empty (mean pixel value > 250)
    if stats.mean[0] > 250:
        return False
        
    # Get bounding box of the actual drawing content
    bbox = Image.eval(img, lambda px: 255-px).getbbox()
    if bbox is None:  # Return false if no content found
        return False
        
    # Calculate content dimensions from bounding box
    width = bbox[2] - bbox[0]   # Width = right - left
    height = bbox[3] - bbox[1]  # Height = bottom - top
    
    # Return false if drawing is too small (< 10x10 pixels)
    if width < 10 or height < 10:
        return False
    
    # Convert the PIL image to a numpy array for more advanced analysis
    # This allows us to perform mathematical operations on the pixel values
    img_array = np.array(img)
    
    # Calculate horizontal and vertical edges using pixel differences
    h_edges = np.diff(img_array, axis=1)
    v_edges = np.diff(img_array, axis=0)
    
    # Count edges with significant intensity change (threshold = 128)
    h_edge_count = np.sum(np.abs(h_edges) > 128)
    v_edge_count = np.sum(np.abs(v_edges) > 128)
    
    # Calculate ratio between horizontal and vertical edges
    edge_ratio = min(h_edge_count, v_edge_count) / max(h_edge_count, v_edge_count)
    
    # Return false if pattern looks like a smiley (balanced edges and complex)
    if edge_ratio > 0.8 and h_edge_count > 1000:
        return False
    
    return True  # All validation checks passed

def preprocess_image(img):
    # Prepare the drawn image for the neural network by converting to MNIST format
    # Return None if input validation fails
    if not is_valid_digit(img):
        return None
    
    # Convert to grayscale and invert colors to match MNIST format
    if img.mode != 'L':
        img = img.convert('L')
    img = ImageOps.invert(img)
    
    # Crop to content bounding box if exists
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Make image square by padding shorter dimension
    width, height = img.size
    if width > height:
        padding = (width - height) // 2
        img = img.crop((-padding, 0, width+padding, width))
    else:
        padding = (height - width) // 2
        img = img.crop((0, -padding, height, height+padding))
    
    # Resize to 20x20 and add padding to reach MNIST size (28x28)
    img = img.resize((20,20), Image.Resampling.LANCZOS)
    padding = (28 - 20) // 2
    img = ImageOps.expand(img, padding)
    
    # Convert to normalized numpy array
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def predict_digit(img):
    # Get model prediction and confidence score for the drawn digit
    processed_img = preprocess_image(img)
    if processed_img is None:
        return -1, 0  # Return error code if preprocessing fails
    
    # Reshape to model input format (1 image, 28x28 pixels, 1 channel)
    processed_img = processed_img.reshape(1, 28, 28, 1)
    
    # Get raw prediction probabilities for each digit (0-9)
    pred = model.predict(processed_img, verbose=0)[0]
    
    # Apply temperature scaling to calibrate confidence (temperature=1.5)
    scaled_pred = np.exp(np.log(pred) / 1.5)
    scaled_pred = scaled_pred / np.sum(scaled_pred)
    
    # Check if prediction confidence meets minimum threshold (30%)
    confidence = np.max(scaled_pred)
    if confidence < 0.3:
        return -1, confidence
    
    return np.argmax(scaled_pred), confidence

class App(tk.Tk):
    def __init__(self):
        # Initialize main window and set title
        tk.Tk.__init__(self)
        self.title("Handwritten Digit Recognizer")
        
        # Set up canvas dimensions and drawing coordinates
        self.x = self.y = 0
        self.canvas_width = 400
        self.canvas_height = 400
        
        # Create drawing canvas with white background and crosshair cursor
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="cross"
        )
        
        # Create display label for showing predictions
        self.label = tk.Label(
            self,
            text="Draw a digit",
            font=("Helvetica", 48)
        )
        
        # Create control buttons for recognition and clearing
        self.classify_btn = tk.Button(
            self,
            text="Recognize",
            command=self.classify_handwriting
        )
        self.button_clear = tk.Button(
            self,
            text="Clear",
            command=self.clear_all
        )
        
        # Create instruction label
        self.instruction = tk.Label(
            self,
            text="Draw a single digit using continuous strokes",
            font=("Helvetica", 12)
        )
        
        # Arrange GUI elements in grid layout
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.instruction.grid(row=2, column=0, columnspan=2, pady=2)
        
        # Bind mouse events for drawing (motion and initial click)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<Button-1>", self.set_point)
        
        # Create blank image for storing the drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def set_point(self, event):
        # Store initial mouse position when drawing starts
        self.x = event.x
        self.y = event.y

    def clear_all(self):
        # Reset canvas, internal image, and display text to initial state
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label.configure(text="Draw a digit")
        
    def classify_handwriting(self):
        # Get prediction and confidence for the drawn digit
        digit, confidence = predict_digit(self.image)
        
        # Handle prediction results
        if digit == -1:
            if confidence == 0:  # Nothing drawn
                self.label.configure(text="Draw something!")
            else:  # Not recognized as digit
                self.label.configure(text="Not a digit")
                self.instruction.configure(text="Try drawing a clearer, single digit")
        else:
            # Show prediction and confidence percentage
            self.label.configure(text=f"{digit}, {int(confidence*100)}%")
            if confidence < 0.5:  # Low confidence
                self.instruction.configure(text="Low confidence - try drawing more clearly")
            else:  # High confidence
                self.instruction.configure(text="Draw a single digit using continuous strokes")

    def draw_lines(self, event):
        # Set brush size for drawing
        r = 12
        
        # Calculate circle bounds at current position
        x1, y1 = (event.x-r), (event.y-r)
        x2, y2 = (event.x+r), (event.y+r)
        
        # Draw circles on both canvas and internal image
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
        
        # Connect to previous point for smooth lines
        if self.x and self.y:
            self.canvas.create_line(
                self.x, self.y,
                event.x, event.y,
                width=r*2,
                fill='black',
                capstyle=tk.ROUND
            )
            self.draw.line(
                [self.x, self.y, event.x, event.y],
                width=r*2,
                fill='black'
            )
        
        # Update previous position
        self.x = event.x
        self.y = event.y

if __name__ == "__main__":
    app = App()
    app.mainloop()