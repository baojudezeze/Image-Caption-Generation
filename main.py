import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import predict


def get_file_path():
    file_path = filedialog.askopenfilename()
    global file_path_predict
    file_path_predict = file_path
    return file_path


# Function to handle image selection
def select_image():
    file_path = get_file_path()
    caption_label.config(text="Caption will appear here...")
    if file_path:
        # Load the image
        image = Image.open(file_path)
        image.thumbnail((500, 500))  # Resize for display
        photo = ImageTk.PhotoImage(image)

        # Display the image
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference!


# Function to generate image caption
def generate_caption():
    caption = predict.generate_caption(file_path_predict)
    caption_label.config(text=caption)
    
    # the Google gTTs voice package
    gTTs.run_gTTs(True, caption)


if __name__ == '__main__':
    # Create the main window
    root = tk.Tk()
    root.title("Image Captioning GUI")
    root.geometry('500x600')

    # Create a label to display the image
    image_label = tk.Label(root)
    image_label.pack()

    # Create a button to select an image
    select_button = tk.Button(root, text="Select Image", command=select_image)
    select_button.pack()

    # Create a label to display the caption
    caption_label = tk.Label(root, text="Caption will appear here...")
    caption_label.pack()

    # Create a button to generate the caption
    generate_button = tk.Button(root, text="Generate Caption", command=generate_caption)
    generate_button.pack()

    # Start the GUI event loop
    root.mainloop()
