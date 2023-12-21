import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from caption import description

class UI(ttk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.title("image_caption")
        self.file_path = None
        
        self.paned_window = tk.PanedWindow(self, orient="horizontal")
        self.paned_window.pack(expand=True, fill="both")

        self.image_label = ttk.Label(self.paned_window)
        self.paned_window.add(self.image_label)
        self.text_box = tk.Text(self.paned_window, wrap="word", width=40, height=20)
        self.paned_window.add(self.text_box)

        self.paned_window.paneconfigure(self.image_label, minsize=600)
        self.paned_window.paneconfigure(self.text_box, minsize=400)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.Label_1 = ttk.Label(self, text="选择图片")
        self.Label_1.pack(side="left", padx=(180, 0), pady=10)

        self.Label_2 = ttk.Label(self, text="描述")
        self.Label_2.pack(side="left", padx=(450, 0), pady=10)

        self.Label_1.bind("<Button-1>",lambda e:self.load_image())
        self.Label_2.bind("<Button-1>",lambda e:self.load_description())

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.file_path:
            image = Image.open(self.file_path)

            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            aspect_ratio = image.width / image.height

            if aspect_ratio > 1:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
            else:
                new_width = int(label_height * aspect_ratio)
                new_height = label_height

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            tk_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image

    def load_description(self):
        if self.file_path:
            result = description(self.file_path)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, result)

if __name__ == "__main__":
    window = UI(themename="cosmo")
    window.geometry("900x650")
    window.mainloop()