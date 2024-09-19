# game-bot/screen_area.py

import tkinter as tk
from PIL import ImageGrab, ImageTk
from screeninfo import get_monitors
from typing import Optional, Tuple

class ScreenAreaSelector:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.cur_x: Optional[int] = None
        self.cur_y: Optional[int] = None

        # Get information about all monitors
        self.monitors = get_monitors()

        # Calculate the total width and height of all monitors
        self.total_width = max(
            monitor.x + monitor.width for monitor in self.monitors)
        self.total_height = max(
            monitor.y + monitor.height for monitor in self.monitors)

        # Set the transparency and geometry of the master window
        self.master.attributes('-alpha', 0.3)
        self.master.geometry(f"{self.total_width}x{self.total_height}+0+0")
        self.master.attributes('-fullscreen', True)
        
        # Bind mouse events to their respective handlers
        self.master.bind('<ButtonPress-1>', self.on_button_press)
        self.master.bind('<B1-Motion>', self.on_move_press)
        self.master.bind('<ButtonRelease-1>', self.on_button_release)

        self.rect: Optional[int] = None

        # Create a canvas to draw the selection rectangle
        self.canvas = tk.Canvas(
            self.master, cursor="cross", width=self.total_width, height=self.total_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def on_button_press(self, event: tk.Event) -> None:
        # Record the starting position of the selection
        self.start_x = int(self.canvas.canvasx(event.x))
        self.start_y = int(self.canvas.canvasy(event.y))

        # If a rectangle already exists, delete it
        if self.rect:
            self.canvas.delete(self.rect)
        
        # Create a new rectangle at the starting position
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event: tk.Event) -> None:
        # Update the current position of the selection
        self.cur_x = int(self.canvas.canvasx(event.x))
        self.cur_y = int(self.canvas.canvasy(event.y))
        
        # Update the coordinates of the rectangle as the mouse moves
        if self.rect and self.start_x is not None and self.start_y is not None:
            self.canvas.coords(self.rect, self.start_x,
                               self.start_y, self.cur_x, self.cur_y)

    def on_button_release(self, event: tk.Event) -> None:
        # Exit the main loop when the mouse button is released
        self.master.quit()

    def get_selection_coordinates(self) -> Tuple[int, int, int, int]:
        # Ensure a rectangle has been drawn
        if self.rect is None:
            raise ValueError("No area selected")
        
        # Get the coordinates of the rectangle
        coords = self.canvas.coords(self.rect)
        
        # Ensure the coordinates are valid
        if len(coords) != 4:
            raise ValueError("Invalid selection coordinates")
        
        return int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

def main() -> None:
    # Create the main Tkinter window
    root = tk.Tk()
    app = ScreenAreaSelector(root)
    root.mainloop()

    try:
        # Get the coordinates of the selected area
        x1, y1, x2, y2 = app.get_selection_coordinates()
        print(f"Selected area: ({x1}, {y1}, {x2}, {y2})")

        # Capture the screenshot across all monitors
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2), all_screens=True)
        screenshot.save("images/game_area.png")
        print("Screenshot saved as game_area.png")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
