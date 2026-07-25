"""Entry point for the Lanczos ED macOS app."""
import sys
import os

# Ensure the package root is on the path when running from a .app bundle
if getattr(sys, 'frozen', False):
    # Running inside PyInstaller bundle
    base = sys._MEIPASS
    os.chdir(os.path.dirname(sys.executable))
else:
    base = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base)

from lanczos_ed.gui.main_window import run_gui
run_gui()
