#Oyunlarda mouse.move ise yaramaz.
import pyautogui

screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = False

center_w = int(screen_width/2)
center_h = int(screen_height/2)
screen_center = (center_w,center_h)

x = center_w
y = center_h

for i in range(50000):
    print(i)

pyautogui.moveRel(1000, 0) # Sağa hareket ettirir
pyautogui.moveRel(0, 1000) # Aşağı hareket ettirir


'''
#Cogu kutuphanenin icinde bu bulunur. Oyunlarda da mouse.move hareketini yapmayi saglar.
import win32api, win32con

for i in range(50):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
    pyautogui.click(x, y)
    x+=10
    #y+=10

'''

'''
#Oyun iclerinde de calisabilir, ancak vanguard gibi sistemler gercek bir usb fare ile mouse.move ozelligini kontrol ettirebilmek icin bu tarz seyleri engeller.
import pydirectinput

pydirectinput.moveRel(300, 0) # Sağa hareket ettirir
pydirectinput.moveRel(0, 300) # Aşağı hareket ettirir
pydirectinput.moveRel(-300, 0) # Sola hareket ettirir
pydirectinput.moveRel(0, -300) # Yukarı hareket ettirir

'''


'''
#Oyun iclerinde de calisabilir, ancak vanguard gibi sistemler gercek bir usb fare ile mouse.move ozelligini kontrol ettirebilmek icin bu tarz seyleri engeller.
from pynput.mouse import Button, Controller
import time

mouse = Controller()

# Read pointer position
print('The current pointer position is {0}'.format(
    mouse.position))

# Move pointer relative to current position
time.sleep(3)
mouse.move(-10000, 2000)
mouse.press(Button.left)
mouse.release(Button.left)
time.sleep(1)
mouse.move(200, -400)

# Set pointer position
mouse.position = (10, 20)
# Press and release
mouse.press(Button.left)
mouse.release(Button.left)

# Double click; this is different from pressing and releasing
# twice on macOS
mouse.click(Button.left, 2)

# Scroll two steps down
mouse.scroll(0, 2)

'''