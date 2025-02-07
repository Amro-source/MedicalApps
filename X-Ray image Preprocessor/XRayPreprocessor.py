# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:08:13 2024

@author: Meshmesh
"""

import wx
import cv2
import numpy as np


class XRayPreprocessorApp(wx.Frame):
    def __init__(self, parent, title):
        super(XRayPreprocessorApp, self).__init__(parent, title=title, size=(800, 600))

        # Panel
        panel = wx.Panel(self)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Title Text
        self.title = wx.StaticText(panel, label="X-Ray Image Preprocessor", style=wx.ALIGN_CENTER)
        font = self.title.GetFont()
        font.PointSize += 10
        font = font.Bold()
        self.title.SetFont(font)
        vbox.Add(self.title, flag=wx.EXPAND | wx.ALL, border=10)

        # Image Display Area
        self.image_display = wx.StaticBitmap(panel)
        vbox.Add(self.image_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Buttons
        button_box = wx.BoxSizer(wx.HORIZONTAL)

        self.load_button = wx.Button(panel, label="Load X-Ray Image")
        self.load_button.Bind(wx.EVT_BUTTON, self.on_load_image)
        button_box.Add(self.load_button, flag=wx.LEFT | wx.RIGHT, border=5)

        self.hist_eq_button = wx.Button(panel, label="Histogram Equalization")
        self.hist_eq_button.Bind(wx.EVT_BUTTON, self.on_histogram_equalization)
        button_box.Add(self.hist_eq_button, flag=wx.LEFT | wx.RIGHT, border=5)

        self.clahe_button = wx.Button(panel, label="Apply CLAHE")
        self.clahe_button.Bind(wx.EVT_BUTTON, self.on_clahe)
        button_box.Add(self.clahe_button, flag=wx.LEFT | wx.RIGHT, border=5)

        self.edge_button = wx.Button(panel, label="Edge Detection")
        self.edge_button.Bind(wx.EVT_BUTTON, self.on_edge_detection)
        button_box.Add(self.edge_button, flag=wx.LEFT | wx.RIGHT, border=5)

        self.denoise_button = wx.Button(panel, label="Noise Reduction")
        self.denoise_button.Bind(wx.EVT_BUTTON, self.on_denoise)
        button_box.Add(self.denoise_button, flag=wx.LEFT | wx.RIGHT, border=5)

        self.save_button = wx.Button(panel, label="Save Processed Image")
        self.save_button.Bind(wx.EVT_BUTTON, self.on_save_image)
        button_box.Add(self.save_button, flag=wx.LEFT | wx.RIGHT, border=5)

        vbox.Add(button_box, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        # Set panel sizer
        panel.SetSizer(vbox)

        # Initialize variables
        self.original_image = None
        self.processed_image = None

        # Show the application
        self.Show()

    def on_load_image(self, event):
        # Open file dialog to select an X-ray image
        file_dialog = wx.FileDialog(self, "Open X-Ray Image", wildcard="Image files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg",
                                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if file_dialog.ShowModal() == wx.ID_CANCEL:
            return  # Cancelled by user

        # Load the selected image
        image_path = file_dialog.GetPath()
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image)
        else:
            wx.MessageBox("Failed to load image. Please select a valid image file.", "Error", wx.ICON_ERROR)

    def display_image(self, image):
        # Convert the image to wx.Bitmap and display it
        height, width = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for wx.Bitmap
        bitmap = wx.Bitmap.FromBuffer(width, height, image_rgb)
        self.image_display.SetBitmap(bitmap)
        self.Refresh()

    def on_histogram_equalization(self, event):
        if self.processed_image is None:
            wx.MessageBox("Please load an X-ray image first!", "Error", wx.ICON_ERROR)
            return
        self.processed_image = cv2.equalizeHist(self.processed_image)
        self.display_image(self.processed_image)

    def on_clahe(self, event):
        if self.processed_image is None:
            wx.MessageBox("Please load an X-ray image first!", "Error", wx.ICON_ERROR)
            return
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(self.processed_image)
        self.display_image(self.processed_image)

    def on_edge_detection(self, event):
        if self.processed_image is None:
            wx.MessageBox("Please load an X-ray image first!", "Error", wx.ICON_ERROR)
            return
        self.processed_image = cv2.Canny(self.processed_image, 100, 200)
        self.display_image(self.processed_image)

    def on_denoise(self, event):
        if self.processed_image is None:
            wx.MessageBox("Please load an X-ray image first!", "Error", wx.ICON_ERROR)
            return
        self.processed_image = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
        self.display_image(self.processed_image)

    def on_save_image(self, event):
        if self.processed_image is None:
            wx.MessageBox("No processed image to save!", "Error", wx.ICON_ERROR)
            return

        file_dialog = wx.FileDialog(self, "Save Processed Image", wildcard="PNG files (*.png)|*.png",
                                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if file_dialog.ShowModal() == wx.ID_CANCEL:
            return  # Cancelled by user

        save_path = file_dialog.GetPath()
        cv2.imwrite(save_path, self.processed_image)
        wx.MessageBox("Processed image saved successfully!", "Info", wx.ICON_INFORMATION)


if __name__ == "__main__":
    app = wx.App(False)
    frame = XRayPreprocessorApp(None, title="X-Ray Image Preprocessor")
    app.MainLoop()
