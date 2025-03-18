import cv2
from threading import *
import win32gui
import win32api
import win32con
from tkinter import filedialog
import tkinter as tk
import json
from tkinter import ttk
import numpy as np


# Class to handle the video frame
class VideoSettings:
    video_window_title = "Selected video"
    frame = None
    frame_Original = None
    cap = None
    cross_point_index = -1
    cross_points = [(0,0), (0,0), (0,0), (0,0)]
    video_size = (640, 360)

    # Read the first video frame from the video
    def GetVideoFrame(self, video_path, frame_clockwise_rotation=0):
        start_frame_number = 1

        # Open the video
        # (When selecting the video, shows in the correct rotation, but when typed rotates)
        self.cap = cv2.VideoCapture(video_path)
        
        # Get total number of frames
        totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Check for valid frame number
        if start_frame_number >= 0 & start_frame_number <= totalFrames:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            ret, self.frame = self.cap.read()

            # Reduce image size
            self.frame = cv2.resize(self.frame, self.video_size)

            # Rotate the frame if required
            if frame_clockwise_rotation == 90:
                self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
            elif frame_clockwise_rotation == 180:
                self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
            elif frame_clockwise_rotation == 270:
                self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
            # Clear all points
            self.ClearAllPoints()

            # Get a copy of the frame
            self.frame_Original = self.frame.copy()

    # Draw a cross of the clicked point.
    def DrawPoint(self, x, y):
        cv2.line(self.frame, (x-10,y), (x+10,y), (0, 0, 255), 1)
        cv2.line(self.frame, (x,y-10), (x,y+10), (0, 0, 255), 1)

    # Clear the last selected point
    def ClearCurrentPoint(self):
        self.cross_points[self.cross_point_index] = (0,0)
        self.cross_point_index = self.cross_point_index - 1

    # Clear all selected points
    def ClearAllPoints(self):
        for point_index in range(0,4):
           self.cross_points[point_index] = (0,0)
        self.cross_point_index = -1
    
    # Function call for the mouse events
    def CaptureMouseEvent(self, event, x, y, flags, params):
        # If the left mouse button is pressed.
        if event == cv2.EVENT_LBUTTONDOWN:
            # If the selected point is less than 3,
            if self.cross_point_index < 3:
                self.cross_point_index = self.cross_point_index + 1
                self.cross_points[self.cross_point_index] = (x,y)
                self.DrawPoint(x, y)
                #cv2.circle(img, (x,y), 5, (0, 255, 255), 4)

                # Draw pedestrian crossing area (Blue transparent overlay)
                if self.cross_point_index == 3:
                    crossing_area = [self.cross_points[0], self.cross_points[1], self.cross_points[3], self.cross_points[2]]
                    overlay = self.frame.copy()
                    cv2.fillPoly(overlay, [np.array(crossing_area, np.int32)], (255, 0, 0))  # Blue color
                    cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)  # Transparency effect

                self.ShowFrame()
        
        # If the right mouse button is pressed.
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.cross_point_index >= 0:
                self.frame = self.frame_Original.copy()
                for point_index in range(0,self.cross_point_index):
                    x, y = self.cross_points[point_index]
                    self.DrawPoint(x, y)
                self.ClearCurrentPoint()
                self.ShowFrame()

    # Show the frame
    def ShowFrame(self):
        cv2.imshow(self.video_window_title, self.frame)

    def ShowVideoFrame(self):
        self.ShowFrame()
        # Set the Mouse Callback function, and call Capture_Event function.
        cv2.setMouseCallback(self.video_window_title, self.CaptureMouseEvent)
        while True:
            whnd_cv1 = self.GetVideoWindowHandle()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or whnd_cv1 == 0:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    # Get the handle of the window
    def GetVideoWindowHandle(self):
        return win32gui.FindWindowEx(None, None, None, self.video_window_title)

    # Close window frame
    def CloseVideoFrame(self):
        whnd_cv = self.GetVideoWindowHandle()
        if not (whnd_cv == 0):
            temp = win32api.PostMessage(whnd_cv, win32con.WM_CHAR, 0x71, 0)


# Main program, global variables
video_settings = VideoSettings()
video_path = ""
frame_rotation = 0
crossing_distance = ""
json_output_file = "video_settings.json"

# Read the path from the textbox
def GetVideoPath():
    global video_path
    video_path = txt_video_path.get(1.0, tk.END)
    return video_path

# Select a video file
def SelectVideo():
    selected_video = filedialog.askopenfilename()
    #txt_video_path.config(state="normal")
    txt_video_path.delete(1.0, tk.END)
    txt_video_path.insert(tk.INSERT, selected_video)
    #txt_video_path.config(state="disabled")
    video_path = GetVideoPath()

# View the video frame in a separate window
def ViewVideoFrame(frame_clockwise_rotation=0):
    video_path = GetVideoPath()
    video_settings.GetVideoFrame(video_path, frame_clockwise_rotation)
    t1 = Thread(target=video_settings.ShowVideoFrame)
    t1.start()

# Rotate the frame
def RotateVideoFrame():
    global frame_rotation
    frame_rotation = frame_rotation + 90
    frame_rotation = frame_rotation % 360
    video_settings.CloseVideoFrame()
    ViewVideoFrame(frame_rotation)

# Read available settigs and close the screen
def ReadSettings():
    global crossing_distance
    #crossing_distance = txt_crossing_distance.get()
    crossing_distance = txt_crossing_distance.get(1.0, tk.END)
    root.quit()
    #root.destroy()


# Defne controls for the UI
root = tk.Tk()
root.title("Capture video attributes")
root.geometry('400x400')

button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=3, pady=3)

btn_video_select = tk.Button(button_frame, text="Select video file", command=SelectVideo)
btn_video_select.grid(row=0, column=0, padx=3, pady=3)

btn_camera_select = tk.Button(button_frame, text="Select camera", state="disabled")
btn_camera_select.grid(row=1, column=0, padx=3, pady=3)

#txt_video_path = tk.Text(root, state='disabled', width=50, height=10)
txt_video_path = tk.Text(root, width=32, height=7)
txt_video_path.grid(row=0, column=1, padx=3, pady=3, columnspan=3)

btn_video_view = tk.Button(root, text="View video first frame", command=ViewVideoFrame)
btn_video_view.grid(row=2, column=0, padx=3, pady=3, columnspan=2, sticky="w")

btn_video_rotate = tk.Button(root, text="Rotate video 90\xb0 clockwise", command=RotateVideoFrame)
btn_video_rotate.grid(row=2, column=2, padx=3, pady=3, columnspan=2)

lbl_sep1 = tk.Label(root, text="-"*80)
lbl_sep1 .grid(row=4, column=0, padx=0, pady=0, columnspan=4)

lbl_1 = tk.Label(root, text="Click on frame in order to mark")
lbl_1.grid(row=5, column=0, padx=0, pady=0, columnspan=4)
lbl_2 = tk.Label(root, text="Left Top, Right Top, Left Down, Right Down")
lbl_2.grid(row=6, column=0, padx=0, pady=0, columnspan=4)
lbl_3 = tk.Label(root, text="points of the pedestrian crossing.")
lbl_3.grid(row=7, column=0, padx=0, pady=0, columnspan=4)
lbl_4 = tk.Label(root, text="Right click to remove last point.")
lbl_4.grid(row=8, column=0, padx=0, pady=0, columnspan=4)

lbl_sep2 = tk.Label(root, text="-"*80)
lbl_sep2.grid(row=9, column=0, padx=0, pady=0, columnspan=4)

lbl_5 = tk.Label(root, text="Enter the distance of the pedestrian crossing in cm")
lbl_5.grid(row=10, column=0, padx=0, pady=0, columnspan=3)

txt_crossing_distance = tk.Text(root, width=7, height=1)
txt_crossing_distance.grid(row=10, column=3, padx=0, pady=0)

lbl_sep3 = tk.Label(root, text="-"*80)
lbl_sep3.grid(row=11, column=0, padx=0, pady=0, columnspan=4)

btn_close = tk.Button(root, text="Read settings and exit", command=ReadSettings)
btn_close.grid(row=12, column=0, padx=0, pady=0, columnspan=4)

root.mainloop()


# When the main screen is closed, close the video frame window as well
video_settings.CloseVideoFrame()

# Remove the enter key from the text inputs
video_path = video_path.replace("\n","")
crossing_distance = crossing_distance.replace("\n","")


# Print settings
print("Video path: ", video_path)
print("Rotation: ", frame_rotation)
print("Crossing points: ")
for point_index in range(4):
    print (video_settings.cross_points[point_index])
print("Distance in cm: ", crossing_distance)
print("Video size: ", video_settings.video_size)


# Write settings to json file
json_string = [{"path": video_path, 
                "rotate": frame_rotation, 
                "top_left":video_settings.cross_points[0], 
                "top_right":video_settings.cross_points[1],
                "bottom_left":video_settings.cross_points[2],
                "bottom_right":video_settings.cross_points[3],
                "distance":crossing_distance,
                "videosize":video_settings.video_size}]

with open(json_output_file, "w", encoding='utf-8') as f:
    json.dump(json_string, f, ensure_ascii=False, indent=4)
