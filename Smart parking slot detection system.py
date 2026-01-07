"""
Smart Parking Slot Detection
Manual Slot Drawing + Persistent Slots (FINAL)
Author: Haran
"""

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading, queue, time, os

VIDEO_PATH = r"D:\projects\DTT PROJECT\Python program\DTT\0000-0203.mp4"
OUTPUT_DIR = os.path.dirname(VIDEO_PATH) or "."
SLOT_CSV = os.path.join(OUTPUT_DIR, "parking_slots_manual.csv")

CANVAS_W = 1000
CANVAS_H = 600
MOTION_RATIO_THRESH = 0.10


class Processor(threading.Thread):
    def __init__(self, video, slots, fq, sq):
        super().__init__(daemon=True)
        self.video = video
        self.slots = slots
        self.fq = fq
        self.sq = sq
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=30, detectShadows=False
        )

    def run(self):
        cap = cv2.VideoCapture(self.video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fg = self.bg.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)

            occupied = []
            for sid,(x1,y1,x2,y2) in enumerate(self.slots,1):
                roi = fg[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if cv2.countNonZero(roi)/roi.size > MOTION_RATIO_THRESH:
                    occupied.append(sid)

            draw = frame.copy()
            for sid,(x1,y1,x2,y2) in enumerate(self.slots,1):
                color = (0,0,255) if sid in occupied else (0,255,0)
                cv2.rectangle(draw,(x1,y1),(x2,y2),color,2)

            if not self.fq.full():
                self.fq.put(draw)
            if not self.sq.full():
                self.sq.put({
                    "occupied": len(occupied),
                    "vacant": len(self.slots)-len(occupied)
                })

            time.sleep(0.03)

        cap.release()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Smart Parking – Persistent Slots")

        self.frame_q = queue.Queue(3)
        self.stat_q = queue.Queue(10)

        self.slots_canvas = []
        self.slots_real = []

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="black")
        self.canvas.pack()

        btns = ttk.Frame(root)
        btns.pack(pady=6)

        ttk.Button(btns, text="Draw Slots", command=self.enable_draw).pack(side="left", padx=5)
        ttk.Button(btns, text="Save Slots", command=self.save_slots).pack(side="left", padx=5)
        ttk.Button(btns, text="Start Simulation", command=self.start_sim).pack(side="left", padx=5)

        self.info = ttk.Label(root, font=("Segoe UI", 12))
        self.info.pack(pady=4)

        self.first_frame = self.load_first_frame()
        self.orig_h, self.orig_w = self.first_frame.shape[:2]

        self.show_image(self.first_frame)

        self.start_x = self.start_y = None
        self.temp_rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.load_saved_slots()

    def load_first_frame(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot read video")
        return frame

    def show_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((CANVAS_W, CANVAS_H))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_img)

    def enable_draw(self):
        self.slots_canvas.clear()
        self.slots_real.clear()
        self.canvas.delete("slot")
        self.info.config(text="Draw slots using mouse")

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="green", width=2, tags="slot"
        )

    def on_drag(self, event):
        if self.temp_rect:
            self.canvas.coords(
                self.temp_rect,
                self.start_x, self.start_y,
                event.x, event.y
            )

    def on_release(self, event):
        if self.temp_rect:
            x1,y1,x2,y2 = self.canvas.coords(self.temp_rect)
            self.slots_canvas.append((x1,y1,x2,y2))
            self.temp_rect = None

    def save_slots(self):
        if not self.slots_canvas:
            messagebox.showwarning("Warning","No slots drawn")
            return

        sx = self.orig_w / CANVAS_W
        sy = self.orig_h / CANVAS_H

        self.slots_real = [
            (int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))
            for (x1,y1,x2,y2) in self.slots_canvas
        ]

        pd.DataFrame(
            self.slots_real,
            columns=["x1","y1","x2","y2"]
        ).to_csv(SLOT_CSV, index=False)

        self.info.config(text=f"{len(self.slots_real)} slots saved")

    def load_saved_slots(self):
        if not os.path.exists(SLOT_CSV):
            return

        df = pd.read_csv(SLOT_CSV)
        self.slots_real = df.values.tolist()

        sx = CANVAS_W / self.orig_w
        sy = CANVAS_H / self.orig_h

        for (x1,y1,x2,y2) in self.slots_real:
            cx1, cy1 = int(x1*sx), int(y1*sy)
            cx2, cy2 = int(x2*sx), int(y2*sy)
            self.slots_canvas.append((cx1,cy1,cx2,cy2))
            self.canvas.create_rectangle(
                cx1,cy1,cx2,cy2,
                outline="green", width=2, tags="slot"
            )

        self.info.config(text=f"{len(self.slots_real)} slots loaded from file")

    def start_sim(self):
        if not self.slots_real:
            messagebox.showwarning("Warning","No slots available")
            return

        self.proc = Processor(
            VIDEO_PATH,
            self.slots_real,
            self.frame_q,
            self.stat_q
        )
        self.proc.start()
        self.update()

    def update(self):
        if not self.frame_q.empty():
            frame = self.frame_q.get()
            self.show_image(frame)

        while not self.stat_q.empty():
            s = self.stat_q.get()
            self.info.config(
                text=f"Occupied: {s['occupied']} | Vacant: {s['vacant']}"
            )

        self.root.after(100, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
