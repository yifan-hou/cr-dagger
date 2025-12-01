import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

class VideoComparisonTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Comparison Tool")
        
        # Variables
        self.video1_path = None
        self.video2_path = None
        self.video1 = None
        self.video2 = None
        self.current_frame_idx = 0
        self.frame1 = None
        self.frame2 = None
        self.tk_frame1 = None
        self.tk_frame2 = None
        self.original_width1 = 0
        self.original_height1 = 0
        self.original_width2 = 0
        self.original_height2 = 0
        self.display_width1 = 400  # Fixed display width for video 1
        self.display_height1 = 300  # Fixed display height for video 1
        self.display_width2 = 400  # Fixed display width for video 2
        self.display_height2 = 300  # Fixed display height for video 2
        
        # Setup UI
        self.setup_ui()
        
        # Bind keys
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<Right>", self.next_frame)
        
    def setup_ui(self):
        # Top frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Load video buttons
        tk.Button(button_frame, text="Load Video 1", command=lambda: self.load_video(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Load Video 2", command=lambda: self.load_video(2)).pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        tk.Button(button_frame, text="Previous Frame (←)", command=lambda: self.prev_frame(None)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Next Frame (→)", command=lambda: self.next_frame(None)).pack(side=tk.LEFT, padx=5)
        
        # Frame counter
        self.frame_counter = tk.Label(button_frame, text="Frame: 0")
        self.frame_counter.pack(side=tk.RIGHT, padx=5)
        
        # Video display frame
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for video 1
        self.canvas1_frame = tk.Frame(self.display_frame)
        self.canvas1_frame.pack(side=tk.LEFT, padx=5)
        self.canvas1 = tk.Canvas(self.canvas1_frame, width=self.display_width1, height=self.display_height1, bg="black")
        self.canvas1.pack()
        
        # Canvas for video 2
        self.canvas2_frame = tk.Frame(self.display_frame)
        self.canvas2_frame.pack(side=tk.LEFT, padx=5)
        self.canvas2 = tk.Canvas(self.canvas2_frame, width=self.display_width2, height=self.display_height2, bg="black")
        self.canvas2.pack()
        
        # Status bar for pixel information
        self.status_bar = tk.Label(self.root, text="Hover over a video to see pixel values", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind mouse motion
        self.canvas1.bind("<Motion>", lambda event: self.show_pixel_info(event, 1))
        self.canvas2.bind("<Motion>", lambda event: self.show_pixel_info(event, 2))
    
    def load_video(self, video_num):
        file_path = filedialog.askopenfilename(
            title=f"Select Video {video_num}",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        if video_num == 1:
            self.video1_path = file_path
            self.video1 = cv2.VideoCapture(self.video1_path)
            self.original_width1 = int(self.video1.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height1 = int(self.video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate appropriate display dimensions while maintaining aspect ratio
            aspect_ratio1 = self.original_width1 / self.original_height1
            self.display_height1 = min(300, int(self.display_width1 / aspect_ratio1))
            self.display_width1 = int(self.display_height1 * aspect_ratio1)
            
            # Update canvas size
            self.canvas1.config(width=self.display_width1, height=self.display_height1)
            
        else:
            self.video2_path = file_path
            self.video2 = cv2.VideoCapture(self.video2_path)
            self.original_width2 = int(self.video2.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height2 = int(self.video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate appropriate display dimensions while maintaining aspect ratio
            aspect_ratio2 = self.original_width2 / self.original_height2
            self.display_height2 = min(300, int(self.display_width2 / aspect_ratio2))
            self.display_width2 = int(self.display_height2 * aspect_ratio2)
            
            # Update canvas size
            self.canvas2.config(width=self.display_width2, height=self.display_height2)
        
        # If both videos are loaded, check if they have the same aspect ratio
        if self.video1 is not None and self.video2 is not None:
            aspect_ratio1 = self.original_width1 / self.original_height1
            aspect_ratio2 = self.original_width2 / self.original_height2
            
            # Allow small floating point differences in aspect ratio (0.5% tolerance)
            if abs(aspect_ratio1 - aspect_ratio2) / aspect_ratio1 > 0.005:
                messagebox.showerror("Error", f"Videos must have the same aspect ratio.\nVideo 1: {aspect_ratio1:.3f}\nVideo 2: {aspect_ratio2:.3f}")
                if video_num == 1:
                    self.video1 = None
                    self.video1_path = None
                else:
                    self.video2 = None
                    self.video2_path = None
                return
            
            # Reset to first frame
            self.current_frame_idx = 0
            self.update_frame()
    
    def update_frame(self):
        if self.video1 is None or self.video2 is None:
            return
        
        # Set position for both videos
        self.video1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.video2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        # Read frames
        ret1, self.frame1 = self.video1.read()
        ret2, self.frame2 = self.video2.read()
        
        if not ret1 or not ret2:
            messagebox.showinfo("End of Video", "Reached the end of one or both videos.")
            if not ret1 and not ret2:
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
            elif not ret1:
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                self.video1.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                _, self.frame1 = self.video1.read()
            else:  # not ret2
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                self.video2.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                _, self.frame2 = self.video2.read()
            return
        
        # Update frame counter
        self.frame_counter.config(text=f"Frame: {self.current_frame_idx}")
        
        # Convert frames to RGB for display
        frame1_rgb = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2RGB)
        
        # Resize frames to fit the fixed display dimensions
        frame1_resized = cv2.resize(frame1_rgb, (self.display_width1, self.display_height1))
        frame2_resized = cv2.resize(frame2_rgb, (self.display_width2, self.display_height2))
        
        # Convert to PhotoImage
        self.tk_frame1 = ImageTk.PhotoImage(image=Image.fromarray(frame1_resized))
        self.tk_frame2 = ImageTk.PhotoImage(image=Image.fromarray(frame2_resized))
        
        # Clear canvas and display new frames
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.tk_frame1)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.tk_frame2)
    
    def next_frame(self, event):
        if self.video1 is None or self.video2 is None:
            return
        
        self.current_frame_idx += 1
        self.update_frame()
    
    def prev_frame(self, event):
        if self.video1 is None or self.video2 is None:
            return
        
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        self.update_frame()
    
    def map_coordinates(self, x, y, from_width, from_height, to_width, to_height):
        """
        Maps coordinates from one resolution to another while maintaining aspect ratio.
        Returns the closest corresponding pixel location.
        """
        # Calculate relative position (0.0 to 1.0)
        relative_x = x / from_width
        relative_y = y / from_height
        
        # Map to the target resolution
        target_x = int(relative_x * to_width)
        target_y = int(relative_y * to_height)
        
        # Ensure we're within bounds
        target_x = max(0, min(to_width - 1, target_x))
        target_y = max(0, min(to_height - 1, target_y))
        
        return target_x, target_y
    
    def show_pixel_info(self, event, source_canvas):
        if self.frame1 is None or self.frame2 is None:
            return
        
        # Get mouse coordinates
        display_x, display_y = event.x, event.y
        
        # Check if coordinates are within display bounds
        if source_canvas == 1:
            if display_x < 0 or display_x >= self.display_width1 or display_y < 0 or display_y >= self.display_height1:
                return
                
            # Map from display coordinates to original frame 1 coordinates
            original_x1, original_y1 = self.map_coordinates(
                display_x, display_y, 
                self.display_width1, self.display_height1, 
                self.original_width1, self.original_height1
            )
            
            # Map from frame 1 coordinates to frame 2 coordinates
            original_x2, original_y2 = self.map_coordinates(
                original_x1, original_y1, 
                self.original_width1, self.original_height1, 
                self.original_width2, self.original_height2
            )
            
            # Map from original frame 2 coordinates to display frame 2 coordinates
            display_x2, display_y2 = self.map_coordinates(
                original_x2, original_y2, 
                self.original_width2, self.original_height2, 
                self.display_width2, self.display_height2
            )
            
        else:  # source_canvas == 2
            if display_x < 0 or display_x >= self.display_width2 or display_y < 0 or display_y >= self.display_height2:
                return
                
            # Map from display coordinates to original frame 2 coordinates
            original_x2, original_y2 = self.map_coordinates(
                display_x, display_y, 
                self.display_width2, self.display_height2, 
                self.original_width2, self.original_height2
            )
            
            # Map from frame 2 coordinates to frame 1 coordinates
            original_x1, original_y1 = self.map_coordinates(
                original_x2, original_y2, 
                self.original_width2, self.original_height2, 
                self.original_width1, self.original_height1
            )
            
            # Map from original frame 1 coordinates to display frame 1 coordinates
            display_x1, display_y1 = self.map_coordinates(
                original_x1, original_y1, 
                self.original_width1, self.original_height1, 
                self.display_width1, self.display_height1
            )
        
        # Get pixel values from original frames
        bgr1 = self.frame1[original_y1, original_x1]
        bgr2 = self.frame2[original_y2, original_x2]
        
        # map pixel to meter
        cm_per_pixel = 200./255.
        cm2 = bgr2 * cm_per_pixel
        # Draw crosshair on the display canvases
        self.draw_crosshair(display_x1 if source_canvas == 2 else display_x, 
                          display_y1 if source_canvas == 2 else display_y, 
                          display_x2 if source_canvas == 1 else display_x, 
                          display_y2 if source_canvas == 1 else display_y)
        
        # Update status bar with pixel information
        self.status_bar.config(text=f"RGB: ({original_x1}, {original_y1}) | "
                               f" Depth: {cm2[2]}")
    
    def draw_crosshair(self, display_x1, display_y1, display_x2, display_y2):
        # Clear previous crosshairs
        self.canvas1.delete("crosshair")
        self.canvas2.delete("crosshair")
        
        # Draw crosshairs on both canvases
        self.canvas1.create_line(0, display_y1, self.display_width1, display_y1, fill="red", tags="crosshair")
        self.canvas1.create_line(display_x1, 0, display_x1, self.display_height1, fill="red", tags="crosshair")
        
        self.canvas2.create_line(0, display_y2, self.display_width2, display_y2, fill="red", tags="crosshair")
        self.canvas2.create_line(display_x2, 0, display_x2, self.display_height2, fill="red", tags="crosshair")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoComparisonTool(root)
    root.geometry("900x600")
    root.mainloop()