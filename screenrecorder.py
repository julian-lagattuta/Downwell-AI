import time
from typing import Tuple
import cv2
import numpy as np
from PIL import Image
from time import sleep
import pygetwindow as gw
import threading
from mss import mss
import cv2
import pyautogui
import pymem
import psutil
from game import State
import torch
import win32gui,win32api
import win32ui
import win32.lib.win32con as win32con 

class DState(State):
    def __init__(self,screen,use_torch=False,device=torch.device("cpu")):
        self.screen = cv2.cvtColor(screen,cv2.COLOR_BGRA2BGR)
        self.gray = cv2.cvtColor(screen,cv2.COLOR_BGRA2GRAY)
        self.feed =cv2.pyrDown(self.gray)
        if use_torch:
            self.feed= torch.tensor(self.feed,device=device).unsqueeze(-1).permute(2,0,1)/255.0

def do_action(self,action):
    self.unpause_game()
    self.press_direction(action)
    time.sleep(self.interval)
    self.unpress_directon(action)

    self.pause_game()
 
class DGame:
    def __init__(self,interval=.2,shell=None,use_torch=False,device=None,pid=None,hwnd=None):
        self.is_paused = False
        self.device = device
        
        self.shell = shell
        pyautogui.PAUSE = 0
        self.screen_dimensions =  (213,120)
        self.hwnd = hwnd
        if hwnd is None:
            self.hwnd = win32gui.FindWindow(None,"Downwell")
        wr = win32gui.GetWindowRect(self.hwnd)
        
        self.depth = 1024
        self.hp = 4
        self.pm = pymem.Pymem(pid)
        self.width= wr[2]-wr[0] 
        self.height = wr[3]-wr[1]

        border_pixels =8 
        titlebar_pixels = 30
        self.width = self.width - (border_pixels * 2)
        self.height = self.height - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        self.offset_x =wr[0] + self.cropped_x
        self.offset_y =wr[1] + self.cropped_y

        if use_torch:
            self.screen_dimensions = (214,120,1)
            self.input_dimensons = (1,214,120)
        self.score = 0
        self.do_pause = False
        self.use_torch = use_torch
        self.pid =pid
        if self.pid is None:
            for proc in psutil.process_iter():
                if "Downwell" in proc.name():
                    self.pid = proc.pid
            if self.pid is None:
                raise Exception("failed to find dowenwll")
        self.process = psutil.Process(self.pid)
        self.interval = interval
    def background_screenshot(self):
        width =self.width
        height = self.height
        hwnd = self.hwnd
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(width, height) , dcObj, (self.cropped_x,self.cropped_y), win32con.SRCCOPY)
        arr = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(arr,dtype="uint8")
        img.shape = (height,width,4)


        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        return img
    def get_temporary_depth(self):

        #
        base = self.pm.base_address
        depth = self.pm.read_ulong(base+0x547A08)
        depth = self.pm.read_ulong(depth+0x24)
        depth = self.pm.read_ulong(depth+0x10)
        depth = self.pm.read_ulong(depth+0x42c)
        depth = self.pm.read_double(depth+0x170)

        return depth
    def get_delta_depth(self):
        cur_depth = self.get_temporary_depth()
        if self.depth-cur_depth>3000:
            self.depth = cur_depth
            return cur_depth
        pdepth = self.depth
        self.depth = max(cur_depth,self.depth) 
        return cur_depth-pdepth
    def get_hp(self):
        path = [0x24,0x10,0x30C]
        base = self.pm.base_address
        hp = self.pm.read_ulong(base+0x547A08)
        for p in path:
            hp = self.pm.read_ulong(hp+p)
        hp = self.pm.read_double(hp+0x3F0)

        return hp
        
    def get_score(self,state:DState):
        base = self.pm.base_address
        score = self.pm.read_ulong(base+0x547A08)   
        score = self.pm.read_ulong(score+0x24)
        score = self.pm.read_ulong(score+0x10)
        score = self.pm.read_ulong(score+0xCB4)
        score = self.pm.read_double(score+0xC0)
        
        return score

        left_x = 100
        bottom_y = 55
        num_arr =[]
        for num, template in number_imgs.items():
            res = cv2.matchTemplate(state.gray[:bottom_y,left_x:], template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            for pt in loc[1]:
                num_arr.append((pt, num))

        num_arr.sort(key=lambda x: x[0])
        if len(num_arr)>0:
            return int(''.join(map(lambda x:str(x[1]),num_arr)))
        raise Exception("failed to get score")
    def pause_game(self):
        if self.is_paused:
            raise Exception("attempted to pause when alreadp aused")
        self.is_paused=True
        self.process.suspend()
    def unpause_game(self):

        if not self.is_paused:
            raise Exception("asdasd")
        self.is_paused=False
        self.process.resume()
    def snap_photo(self):

        return self.background_screenshot()
    def restart(self):
        win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,0x1B,0)
        sleep(.1)
        win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,0x1B,0)
        sleep(.1)
        win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,0x27,0)
        sleep(.1)
        win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,0x27,0)
        sleep(.1)
        for i in range(2):
            win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,0x20,0)
            sleep(.1)
            win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,0x20,0)
            sleep(.1)
        self.score = 0
        self.depth = 1024
    def get_state(self):
        return DState(self.snap_photo(),use_torch=self.use_torch,device=self.device)

    def press_direction(self,action):
    
        if action==5:
            return
            
        if action<3:
            win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,[0x25,0x27,0x20][action],0)
            return
        win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,0x20,0)
        win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,[0x25,0x27][action-3],0)
    def unpress_directon(self,action):
        if action==5:
            return
        if action<3:
            win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,[0x25,0x27,0x20][action],0)
            return
        win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,0x20,0)
        win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,[0x25,0x27][action-3],0)
    def is_loss(self,state:DState):
        right_x = 70
        bottom_y = 52
        threshold = .8
        result= cv2.matchTemplate(state.gray[:bottom_y,:right_x],game_over_img,cv2.TM_CCOEFF_NORMED)
        loc = np.where(result>=threshold)
        if len(loc[0])>0:
            return True
        return False
    def calculate_reward(self,state:DState)-> Tuple[int,bool]:
        prev_score = self.score
        self.score = self.get_score(state)
        new_hp = self.get_hp()
        reward = self.score-prev_score
        reward/=10
        reward = min(.6,reward)
        if reward<0:
            reward =1
        reward+=(new_hp-self.hp)*3
        self.hp = new_hp

        delta_depth=self.get_delta_depth()/(32*3/2) #2/3rd blocks
        reward+=max(0,delta_depth)
        is_loss = self.is_loss(state)

        

        if reward==0:
            reward = -.1
        if is_loss:
            self.depth =1024 
            self.score = 0
            self.hp = 4
        return reward,is_loss
    def move(self,action)->Tuple[DState,int,bool,threading.Thread]:

        state = self.get_state()
        reward,is_loss = self.calculate_reward(state)
        if action!=-1:
            t = threading.Thread(target=do_action,args=(self,action))
            t.start()
        else:
            t = threading.Thread(target=lambda x:x,args=(1,))
            t.start()

        return state,reward,is_loss,t
    def restart_loss(self):
        for i in range(10):
            win32api.SendMessage(self.hwnd,win32con.WM_KEYDOWN,0x20,0)
            sleep(.1)
            win32api.SendMessage(self.hwnd,win32con.WM_KEYUP,0x20,0)
            sleep(0.3)
        sleep(2)
number_img_names = ["font/0.png","font/1.png","font/2.png","font/3.png","font/4.png","font/5.png","font/6.png","font/7.png","font/8.png","font/9.png"]
number_imgs = {i:cv2.imread(v,0) for i,v in enumerate(number_img_names)}
number_imgs["slash"] = cv2.imread("font/slash.png",0)
game_over_img = cv2.imread("font/game_over.png",0)
