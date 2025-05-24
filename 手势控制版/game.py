import pygame
import random
import sys
import json
import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer
import math

def recognize_gesture(hand_landmarks):
    """识别手势类型"""
    # 获取所有手指的关键点
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # 获取所有手指的MCP（掌指关节）点
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # 检查每个手指是否弯曲
    def is_finger_bent(tip, mcp):
        """检查手指是否弯曲，考虑手指可能指向不同方向"""
        # 计算指尖和MCP点之间的向量
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        
        # 计算手指指向的方向
        angle = math.atan2(dy, dx)
        
        # 根据手指指向的方向判断是否弯曲
        # 将角度转换为0-360度
        angle_degrees = math.degrees(angle) % 360
        
        # 定义不同方向的判定范围
        # 向上: 315-45度
        # 向右: 45-135度
        # 向下: 135-225度
        # 向左: 225-315度
        
        # 根据方向判断是否弯曲（返回True表示弯曲，False表示伸直）
        if 315 <= angle_degrees or angle_degrees < 45:  # 向上
            return tip.y > mcp.y  # 指尖在MCP点下方时判定为弯曲
        elif 45 <= angle_degrees < 135:  # 向右
            return tip.x < mcp.x  # 指尖在MCP点左侧时判定为弯曲
        elif 135 <= angle_degrees < 225:  # 向下
            return tip.y < mcp.y  # 指尖在MCP点上方时判定为弯曲
        else:  # 向左
            return tip.x > mcp.x  # 指尖在MCP点右侧时判定为弯曲
    
    # 检查所有手指是否弯曲
    thumb_bent = is_finger_bent(thumb_tip, thumb_mcp)
    index_bent = is_finger_bent(index_tip, index_mcp)
    middle_bent = is_finger_bent(middle_tip, middle_mcp)
    ring_bent = is_finger_bent(ring_tip, ring_mcp)
    pinky_bent = is_finger_bent(pinky_tip, pinky_mcp)
    
    # 计算伸直的手指数量
    extended_fingers = sum([not thumb_bent, not index_bent, not middle_bent, not ring_bent, not pinky_bent])
    
    # 根据伸直的手指数量判断手势
    if extended_fingers <= 1:  # 0或1个手指伸直时视为拳头
        return "拳头"
    elif extended_fingers == 5:  # 所有手指都伸直
        return "手掌"
    elif extended_fingers in [2, 3]:  # 2-3根手指伸直
        # 获取食指和中指的指尖位置
        extended_tips = []
        if not index_bent:
            extended_tips.append(index_tip)
        if not middle_bent:
            extended_tips.append(middle_tip)
        if not ring_bent:
            extended_tips.append(ring_tip)
        
        if len(extended_tips) >= 2:
            # 计算伸直手指的平均位置
            avg_x = sum(tip.x for tip in extended_tips) / len(extended_tips)
            avg_y = sum(tip.y for tip in extended_tips) / len(extended_tips)
            
            # 计算手指指向的方向
            # 使用食指和中指的MCP点作为参考点
            ref_x = (index_mcp.x + middle_mcp.x) / 2
            ref_y = (index_mcp.y + middle_mcp.y) / 2
            
            # 计算方向向量
            dx = avg_x - ref_x
            dy = avg_y - ref_y
            
            # 降低方向判断的阈值，使识别更容易
            direction_threshold = 0.05  # 降低阈值，使方向判断更宽松
            
            # 判断指向方向
            if abs(dx) > direction_threshold or abs(dy) > direction_threshold:  # 只要有一个方向超过阈值就判断
                if abs(dx) > abs(dy):  # 水平方向更明显
                    if dx > 0:
                        return "指向右"
                    else:
                        return "指向左"
                else:  # 垂直方向更明显
                    if dy > 0:
                        return "指向下"
                    else:
                        return "指向上"
    
    return "其他"  # 其他手势状态

def update_selected_area(game_state, screen_x, screen_y):
    """根据手势位置更新选中的区域，并实现吸附效果"""
    # 如果正在拖放，更新拖放位置
    if game_state.dragging and game_state.selected_material:
        # 计算考虑偏移量的拖放位置
        game_state.drag_pos = (screen_x, screen_y)
        return
    
    # 限制手部移动范围
    MAX_X = BOARD_X + BOARD_SIZE * CELL_SIZE  # 限制到棋盘右侧
    MAX_Y = BOARD_Y + BOARD_SIZE * CELL_SIZE  # 限制到棋盘底部
    screen_x = max(LEFT_BUTTONS_X, min(screen_x, MAX_X))
    screen_y = max(LEFT_BUTTONS_START_Y, min(screen_y, MAX_Y))
    
    # 更新实际手部位置（用于显示蓝色小圆圈）
    game_state.hand_position = (screen_x, screen_y)
    
    # 平滑处理手部移动
    if game_state.last_hand_position is None:
        game_state.last_hand_position = (screen_x, screen_y)
    
    # 使用线性插值平滑移动
    smooth_x = game_state.last_hand_position[0] + (screen_x - game_state.last_hand_position[0]) * game_state.smooth_factor
    smooth_y = game_state.last_hand_position[1] + (screen_y - game_state.last_hand_position[1]) * game_state.smooth_factor
    
    # 更新上一次位置
    game_state.last_hand_position = (smooth_x, smooth_y)
    
    # 使用平滑后的坐标
    screen_x, screen_y = int(smooth_x), int(smooth_y)
    
    # 限制手部坐标在有效交互区域内
    def clamp_coordinates(x, y):
        # 定义有效区域
        valid_areas = [
            # 棋盘区域
            (BOARD_X - 100, BOARD_Y, BOARD_X + BOARD_SIZE * CELL_SIZE + 100, BOARD_Y + BOARD_SIZE * CELL_SIZE),
        ]
        
        # 检查是否在任何有效区域内
        for x1, y1, x2, y2 in valid_areas:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return x, y
        
        # 如果不在任何有效区域内，找到最近的有效区域
        min_dist = float('inf')
        closest_x, closest_y = x, y
        
        for x1, y1, x2, y2 in valid_areas:
            # 计算到区域边界的距离
            dist_x = min(abs(x - x1), abs(x - x2))
            dist_y = min(abs(y - y1), abs(y - y2))
            dist = min(dist_x, dist_y)
            
            if dist < min_dist:
                min_dist = dist
                # 将坐标限制在区域内
                closest_x = max(x1, min(x, x2))
                closest_y = max(y1, min(y, y2))
        
        return closest_x, closest_y
    
    # 计算最近的吸附点
    def get_snap_point(x, y):
        # 定义吸附灵敏度
        SNAP_THRESHOLD = 50  # 增大吸附范围
        # 棋盘区域（放在最后检查，因为优先级最低）
        if (BOARD_X <= x <= BOARD_X + BOARD_SIZE * CELL_SIZE and 
            BOARD_Y <= y <= BOARD_Y + BOARD_SIZE * CELL_SIZE):
            col = round((x - BOARD_X) / CELL_SIZE)
            row = round((y - BOARD_Y) / CELL_SIZE)
            # 检查是否在吸附范围内
            if abs(x - (BOARD_X + col * CELL_SIZE)) <= SNAP_THRESHOLD and \
               abs(y - (BOARD_Y + row * CELL_SIZE)) <= SNAP_THRESHOLD:
                return (BOARD_X + col * CELL_SIZE, BOARD_Y + row * CELL_SIZE), (row, col), (BOARD_X + col * CELL_SIZE, BOARD_Y + row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        
        return None, None, None
    
    # 限制手部坐标
    screen_x, screen_y = clamp_coordinates(screen_x, screen_y)
    
    # 获取最近的吸附点
    snap_point, area_info, snap_rect = get_snap_point(screen_x, screen_y)
    
    if snap_point:
        # 更新选中的区域
        game_state.selected_area = area_info
        # 更新吸附区域信息
        game_state.snap_rect = snap_rect
    else:
        game_state.selected_area = None
        game_state.snap_rect = None

def handle_gesture_click(game_state, board, screen_x, screen_y):
    """处理手势点击事件"""
    if game_state.selected_area is None:
        return
        
    if isinstance(game_state.selected_area, tuple):
        if game_state.selected_area[0] == "left_button":
            # 处理左侧按钮点击
            button_index = game_state.selected_area[1]
            if button_index < len(game_state.materials):
                game_state.selected_material = game_state.materials[button_index]
                game_state.selected_material_index = button_index
                game_state.dragging = True  # 启用手势拖放
                game_state.drag_pos = (screen_x, screen_y)  # 设置初始拖放位置
                # 计算材料的中心点偏移
                center_i, center_j = game_state.selected_material.get_center()
                game_state.drag_offset_x = center_j * CELL_SIZE
                game_state.drag_offset_y = center_i * CELL_SIZE
                game_state.message = "已选择材料"
                game_state.message_timer = pygame.time.get_ticks()

    if game_state.current_gesture == "指向左" and game_state.selected_material:
        # 左旋转材料
        game_state.selected_material.rotate(clockwise=False)
        game_state.message = "逆时针旋转"
        game_state.message_timer = pygame.time.get_ticks()
    elif game_state.current_gesture == "指向右":
        # 撤销操作
        if game_state.undo_stack:
            last_state = game_state.undo_stack.pop()
            # 恢复棋盘状态
            board.grid = [row[:] for row in last_state['grid']]
            board.forbidden = set(last_state['forbidden'])
            # 恢复材料栏
            game_state.materials.clear()
            for mat_data in last_state['materials']:
                if len(game_state.materials) < game_state.max_materials:
                    mat = Material(mat_data['size'])
                    mat.cells = [(int(i), int(j)) for i, j in mat_data['cells']]
                    mat.joints = mat_data['joints']
                    mat.color = mat_data['color']
                    mat.special_place = mat_data['special_place']
                    game_state.materials.append(mat)
            # 恢复分数
            game_state.score = last_state['score']
            # 恢复可用材料次数
            game_state.available_materials = last_state['available_materials']
            # 显示提示信息
            game_state.message = "已撤销上一步"
        else:
            game_state.message = "没有可撤销的操作"
        game_state.message_timer = pygame.time.get_ticks()
    elif game_state.current_gesture == "指向上":
        # 使用黄色能力
        if game_state.yellow_ability > 0:
            game_state.yellow_ability -= 1
            game_state.message = "黄色能力！点击一个材料使其变为全联结点"
            game_state.active_ability = "yellow"
        else:
            game_state.message = "黄色能力次数不足"
        game_state.message_timer = pygame.time.get_ticks()
    elif game_state.current_gesture == "指向下":
        # 使用紫色能力
        if game_state.purple_ability > 0:
            game_state.purple_ability -= 1
            game_state.message = "紫色能力！下一个放置的材料将覆盖原有区域"
            game_state.active_ability = "purple"
        else:
            game_state.message = "紫色能力次数不足"
        game_state.message_timer = pygame.time.get_ticks()

def update_connections_and_eliminations(game_state, board):
    """更新联结数并检查消除，并直接触发红色和绿色能力"""
    # 计算联结数
    game_state.connections = 0
    for i in range(board.size):
        for j in range(board.size):
            if board.grid[i][j] and board.grid[i][j]['is_joint']:
                current_color = board.grid[i][j]['color']
                # 检查右侧同色联结点
                if j + 1 < board.size and board.grid[i][j + 1] and \
                   board.grid[i][j + 1]['is_joint'] and board.grid[i][j + 1]['color'] == current_color:
                    game_state.connections += 1
                # 检查下方同色联结点
                if i + 1 < board.size and board.grid[i + 1][j] and \
                   board.grid[i + 1][j]['is_joint'] and board.grid[i + 1][j]['color'] == current_color:
                    game_state.connections += 1

    # 检查行消除
    for i in range(board.size):
        row_color = None
        full_row = True
        for j in range(board.size):
            if (i, j) in board.forbidden:
                continue
            if not board.grid[i][j]:
                full_row = False
                break
            if row_color is None:
                row_color = board.grid[i][j]['color']
            elif board.grid[i][j]['color'] != row_color:
                full_row = False
                break
        if full_row:
            # 消除该行
            for j in range(board.size):
                if (i, j) not in board.forbidden:
                    board.grid[i][j] = None
            game_state.elimination_count += 1
            # 根据颜色触发能力
            if row_color == RED:
                # 红色能力：直接加分
                bonus = board.size * 2  # 例如，每行消除加 (棋盘大小 x 2) 分
                game_state.score += bonus
                game_state.message = f"红色能力！获得{bonus}分"
            elif row_color == GREEN:
                # 绿色能力：直接补充材料
                new_mats = min(3, game_state.max_materials - len(game_state.materials))
                game_state.materials.extend([Material(random.randint(1, 5)) for _ in range(new_mats)])
                game_state.message = "绿色能力！获得3个新材料"
            elif row_color == YELLOW:
                # 黄色能力：储存次数
                game_state.yellow_ability += 1
                game_state.message = "黄色能力次数+1"
            elif row_color == PURPLE:
                # 紫色能力：储存次数
                game_state.purple_ability += 1
                game_state.message = "紫色能力次数+1"
            game_state.message_timer = pygame.time.get_ticks()

    # 检查列消除（逻辑与行消除相同）
    for j in range(board.size):
        col_color = None
        full_col = True
        for i in range(board.size):
            if (i, j) in board.forbidden:
                continue
            if not board.grid[i][j]:
                full_col = False
                break
            if col_color is None:
                col_color = board.grid[i][j]['color']
            elif board.grid[i][j]['color'] != col_color:
                full_col = False
                break
        if full_col:
            # 消除该列
            for i in range(board.size):
                if (i, j) not in board.forbidden:
                    board.grid[i][j] = None
            game_state.elimination_count += 1
            # 根据颜色触发能力
            if col_color == RED:
                bonus = board.size * 2
                game_state.score += bonus
                game_state.message = f"红色能力！获得{bonus}分"
            elif col_color == GREEN:
                new_mats = min(3, game_state.max_materials - len(game_state.materials))
                game_state.materials.extend([Material(random.randint(1, 5)) for _ in range(new_mats)])
                game_state.message = "绿色能力！获得3个新材料"
            elif col_color == YELLOW:
                game_state.yellow_ability += 1
                game_state.message = "黄色能力次数+1"
            elif col_color == PURPLE:
                game_state.purple_ability += 1
                game_state.message = "紫色能力次数+1"
            game_state.message_timer = pygame.time.get_ticks()

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 初始化pygame
pygame.init()
pygame.mixer.init()  # 初始化音频系统

# 加载并播放背景音乐
try:
    pygame.mixer.music.load("music.flac")
    pygame.mixer.music.play(-1)  # -1表示循环播放
except:
    print("无法加载音乐文件")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
COLORS = [RED, GREEN, YELLOW, PURPLE]

# 材料类
class Material:
    def __init__(self, size):
        self.special_place = False  # 是否可以覆盖放置
        self.size = size
        self.cells = []
        self.color = random.choice(COLORS)
        self._generate_shape()
        self._add_joints()
    
    def _generate_shape(self):
        # 根据大小生成不同形状
        if self.size == 1:
            self.cells = [(0, 0)]
        elif self.size == 2:
            self.cells = [(0, 0), (0, 1)]  
        elif self.size == 3:
            if random.random() < 0.5:  # 直线
                self.cells = [(0, 0), (0, 1), (0, 2)]  
            else:  # L形
                self.cells = [(0, 0), (0, 1), (1, 1)]
        elif self.size == 4:
            shape_type = random.choice(["L", "T", "cross","square"])
            if shape_type == "L":  # L形
                self.cells = [(0, 0), (0, 1), (0, 2), (1, 2)]
            elif shape_type == "T":  # T形
                self.cells = [(0, 0), (0, 1), (0, 2), (1, 1)]
            elif shape_type == "cross":  
                self.cells = [(0, 0), (1, 0), (1, 1), (2, 1)]
            elif shape_type == "square":  # 正方形
                self.cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:  # size == 5
            shape_type = random.choice(["trapezoid", "L", "cross", "T"])
            if shape_type == "trapezoid":
                self.cells = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]
            elif shape_type == "L":
                self.cells = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
            elif shape_type == "cross":
                self.cells = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
            else:  # T形
                self.cells = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
    
    def _add_joints(self):
        # 50%概率将格点变为联结点
        self.joints = [random.random() < 0.5 for _ in range(self.size)]
    
    def rotate(self, clockwise=True):
        """绕材料中心精确旋转，确保坐标对齐"""
        # 计算材料的边界
        min_i, max_i, min_j, max_j = self.get_bounds()
        width = max_j - min_j + 1
        height = max_i - min_i + 1
        
        # 创建新的单元格列表
        new_cells = []
        for (i, j) in self.cells:
            # 将坐标转换为相对于左上角的位置
            rel_i = i - min_i
            rel_j = j - min_j
            
            # 执行90度旋转
            if clockwise:
                new_i = rel_j
                new_j = height - 1 - rel_i
            else:
                new_i = width - 1 - rel_j
                new_j = rel_i
            
            # 添加新的坐标
            new_cells.append((new_i, new_j))
        
        # 更新旋转后的单元格
        self.cells = new_cells
        # 保持联结点状态不变
    
    def draw(self, surface, x, y, cell_size):
        # 绘制材料
        for idx, (i, j) in enumerate(self.cells):
            rect = pygame.Rect(
                x + j * cell_size,
                y + i * cell_size,
                cell_size, cell_size
            )
            color = self.color
            if self.joints[idx]:
                # 联结点颜色更亮，增加对比度
                color = tuple(min(c + 80, 255) for c in color)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, WHITE, rect, 1)
            # 在联结点中心绘制反色空心圆
            if self.joints[idx]:
                center = (rect.x + rect.width//2, rect.y + rect.height//2)
                radius = rect.width//4
                border_color = tuple(255 - c for c in color)
                pygame.draw.circle(surface, border_color, center, radius, 2)

    def get_bounds(self):
        """获取材料的边界框"""
        min_i = min(i for i, _ in self.cells)
        max_i = max(i for i, _ in self.cells)
        min_j = min(j for _, j in self.cells)
        max_j = max(j for _, j in self.cells)
        return min_i, max_i, min_j, max_j

    def get_center(self):
        """获取材料的中心点"""
        min_i, max_i, min_j, max_j = self.get_bounds()
        return ((min_i + max_i) / 2, (min_j + max_j) / 2)

# 游戏设置
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 800
BOARD_SIZE = 7  # 初始棋盘大小
CELL_SIZE = 40  # 每个格子大小
BOARD_X = (SCREEN_WIDTH - BOARD_SIZE * CELL_SIZE) // 2  # 居中棋盘
BOARD_Y = 150  # 棋盘y坐标
MATERIAL_START_X = 50
MATERIAL_SPACING = 180  # 材料之间的间距
MATERIAL_AREA_Y = 450
MATERIAL_SIZE = 30
UI_AREA_HEIGHT = 130
# 添加左侧材料按钮的位置设置
LEFT_BUTTON_SIZE = 40  # 按钮大小
LEFT_BUTTONS_X = 350  # 左侧按钮区的起始x坐标
LEFT_BUTTONS_START_Y = 150  # 左侧按钮区的起始y坐标
LEFT_BUTTON_SPACING = 60  # 按钮之间的间距

# 游戏状态
class GameState:
    def __init__(self):
        self.score = 0
        self.connections = 0
        self.last_reward_connection = 0
        self.available_materials = 15
        self.max_materials = 5
        self.elimination_count = 0
        self.selected_material = None
        self.selected_material_index = None
        self.dragging = False
        self.drag_pos = (0, 0)
        self.long_press_timer = 0
        self.message = ""
        self.message_timer = 0
        self.red_ability = 0
        self.green_ability = 0
        self.yellow_ability = 0  
        self.purple_ability = 0
        self.undo_stack = []
        self.animation_frames = []
        self.animation_timer = 0
        self.animation_duration = 300
        self.animation_active = False
        self.music_playing = True
        self.active_ability = None
        self.is_endless_mode = False
        # 修复：将 self.board 初始化为 Board 类的实例
        self.board = Board(BOARD_SIZE)  # 替换原来的列表初始化
        self.current_material = None
        self.current_material_pos = [0, 0]
        self.current_material_rotation = 0
        self.materials = []  # 初始化材料栏为空
        self.game_over = False
        self.selected_area = None
        self.is_gesture_control = False
        self.gesture_click_cooldown = 800
        self.last_gesture_click_time = 0
        self.current_gesture = "等待手势"
        self.previous_gesture = "等待手势"
        self.hand_position = None
        self.snap_rect = None
        self.last_hand_position = None
        self.smooth_factor = 0.5
        self.hand_indicator_radius = 5
        self.help_dialog = HelpDialog()

        # 自动补齐材料栏（不占用剩余材料）
        while len(self.materials) < self.max_materials:
            self.materials.append(Material(random.randint(1, 5)))
    
    def save(self):
        return {
            'score': self.score,
            'connections': self.connections,
            'last_reward_connection': self.last_reward_connection,
            'available_materials': self.available_materials,
            'elimination_count': self.elimination_count,
            'red_ability': self.red_ability,
            'green_ability': self.green_ability,
            'yellow_ability': self.yellow_ability,
            'purple_ability': self.purple_ability
        }
    
    def load(self, data):
        self.score = data.get('score', 0)
        self.connections = data.get('connections', 0)
        self.last_reward_connection = data.get('last_reward_connection', 0)
        self.available_materials = data.get('available_materials', 15)
        self.elimination_count = data.get('elimination_count', 0)
        self.red_ability = data.get('red_ability', 3)
        self.green_ability = data.get('green_ability', 3)
        self.yellow_ability = data.get('yellow_ability', 3)
        self.purple_ability = data.get('purple_ability', 3)

    def reset(self):
        """重置游戏状态"""
        self.score = 0
        self.connections = 0
        self.last_reward_connection = 0
        self.available_materials = 15
        self.elimination_count = 0
        self.selected_material = None
        self.selected_material_index = None
        self.dragging = False
        self.drag_pos = (0, 0)
        self.long_press_timer = 0
        self.message = "游戏已重新开始"
        self.message_timer = pygame.time.get_ticks()
        self.red_ability = 0
        self.green_ability = 0
        self.yellow_ability = 0
        self.purple_ability = 0
        self.undo_stack = []
        self.animation_frames = []
        self.animation_timer = 0
        self.animation_active = False
        self.active_ability = None
        self.is_endless_mode = False
        self.board = Board(BOARD_SIZE)
        self.current_material = None
        self.current_material_pos = [0, 0]
        self.current_material_rotation = 0
        self.materials = []
        self.game_over = False
        self.selected_area = None
        self.is_gesture_control = False
        self.gesture_click_cooldown = 800  # 重置点击灵敏度
        self.last_gesture_click_time = 0
        self.current_gesture = "等待手势"
        self.previous_gesture = "等待手势"
        self.hand_position = None
        self.snap_rect = None
        self.last_hand_position = None
        self.smooth_factor = 0.5  # 重置移动灵敏度
        self.hand_indicator_radius = 5
        self.help_dialog = HelpDialog()  # 重置help_dialog

    def update_gesture(self, gesture):
        """更新手势状态"""
        self.previous_gesture = self.current_gesture
        self.current_gesture = gesture
        
        if gesture == "指向右":
            if self.undo_stack:
                last_state = self.undo_stack.pop()
                self.board.grid = [row[:] for row in last_state['grid']]
                self.board.forbidden = set(last_state['forbidden'])
                # 恢复材料栏
                self.materials.clear()
                for mat_data in last_state['materials']:
                    if len(self.materials) < self.max_materials:
                        mat = Material(mat_data['size'])
                        mat.cells = [(int(i), int(j)) for i, j in mat_data['cells']]
                        mat.joints = mat_data['joints']
                        mat.color = mat_data['color']
                        mat.special_place = mat_data['special_place']
                        self.materials.append(mat)
                # 恢复分数
                self.score = last_state['score']
                # 恢复可用材料次数
                self.available_materials = last_state['available_materials']
                # 显示提示信息
                self.message = "已撤销上一步"
            else:
                self.message = "没有可撤销的操作"
            self.message_timer = pygame.time.get_ticks()

    def is_click(self):
        # 当手势从其他状态变为拳头时，视为点击
        return (self.previous_gesture != "拳头" and 
                self.current_gesture == "拳头")
    
    def can_click(self, current_time):
        # 检查是否可以进行点击（考虑冷却时间）
        if current_time - self.last_gesture_click_time >= self.gesture_click_cooldown:
            self.last_gesture_click_time = current_time
            return True
        return False

# 字体
font = pygame.font.SysFont("SimHei", 24)
help_font = pygame.font.SysFont("SimHei", 20)

# 帮助弹窗类
class HelpDialog:
    def __init__(self):
        self.visible = False
        self.lines = [
            "拖动材料到棋盘上放置",
            "当材料充满一行或一列时，会自动消除。",
            "记得提前旋转材料，及时补充材料和使用能力哦！",
            "努力填满棋盘以获得高分吧！100分即可胜利！",
            "注意：点击重新开始结算游戏；",
            "      获得新材料后撤销也会消耗材料次数",
            "点击弹窗外任意位置关闭",
            "手势控制指南：拳头代表点击",
            "伸出两三根手指向左旋转材料，向右撤销，",
            "指向上方使用黄色能力，指向下方使用紫色能力",
            "在棋盘上移动并点击即可放置材料"
        ]
    
    def toggle(self):
        self.visible = not self.visible
        
    def draw(self, surface):
        if not self.visible:
            return
            
        # 半透明背景
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        surface.blit(s, (0, 0))
        
        # 弹窗背景
        dialog_width = 500
        dialog_height = 400
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(surface, (250, 250, 250), 
                        (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(surface, (200, 200, 200), 
                        (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        
        # 绘制标题
        title = font.render("游戏帮助", True, (0, 0, 0))
        surface.blit(title, (dialog_x + dialog_width//2 - title.get_width()//2, dialog_y + 10))
        
        # 绘制帮助文本
        for i, line in enumerate(self.lines):
            text = help_font.render(line, True, (0, 0, 0))
            surface.blit(text, (dialog_x + 20, dialog_y + 40 + i * 25))

class GameOverDialog:
    def __init__(self):
        self.visible = False
        self.score = 0
        self.connections = 0
        self.elimination_count = 0
        self.is_victory = False  # 添加胜利标志
    
    def show(self, score, connections, elimination_count, is_victory=False):
        self.visible = True
        self.score = score
        self.connections = connections
        self.elimination_count = elimination_count
        self.is_victory = is_victory
    
    def hide(self):
        self.visible = False
        
    def draw(self, surface):
        if not self.visible:
            return
            
        # 半透明背景
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        surface.blit(s, (0, 0))
        
        # 弹窗背景
        dialog_width = 400
        dialog_height = 300
        dialog_x = (SCREEN_WIDTH - dialog_width) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(surface, (250, 250, 250), 
                        (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(surface, (200, 200, 200), 
                        (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        
        # 绘制标题
        title = font.render("游戏胜利" if self.is_victory else "游戏结算", True, (0, 0, 0))
        surface.blit(title, (dialog_x + dialog_width//2 - title.get_width()//2, dialog_y + 20))
        
        # 绘制结算信息
        score_text = font.render(f"最终得分: {self.score}", True, (0, 0, 0))
        conn_text = font.render(f"总联结数: {self.connections}", True, (0, 0, 0))
        elim_text = font.render(f"消除次数: {self.elimination_count}", True, (0, 0, 0))
        
        surface.blit(score_text, (dialog_x + dialog_width//2 - score_text.get_width()//2, dialog_y + 80))
        surface.blit(conn_text, (dialog_x + dialog_width//2 - conn_text.get_width()//2, dialog_y + 120))
        surface.blit(elim_text, (dialog_x + dialog_width//2 - elim_text.get_width()//2, dialog_y + 160))
        
        # 绘制按钮
        button_width = 120
        button_height = 40
        button_spacing = 20
        
        # 继续游戏按钮（仅在胜利时显示）
        if self.is_victory:
            continue_button = pygame.Rect(dialog_x + dialog_width//2 - button_width - button_spacing//2, 
                                        dialog_y + dialog_height - 80, 
                                        button_width, button_height)
            pygame.draw.rect(surface, (200, 200, 200), continue_button)
            pygame.draw.rect(surface, (0, 0, 0), continue_button, 2)
            continue_text = font.render("继续游戏", True, (0, 0, 0))
            surface.blit(continue_text, (continue_button.centerx - continue_text.get_width()//2, 
                                       continue_button.centery - continue_text.get_height()//2))
        
        # 重新开始按钮
        restart_button = pygame.Rect(dialog_x + dialog_width//2 + button_spacing//2 if self.is_victory else dialog_x + dialog_width//2 - button_width//2, 
                                   dialog_y + dialog_height - 80, 
                                   button_width, button_height)
        pygame.draw.rect(surface, (200, 200, 200), restart_button)
        pygame.draw.rect(surface, (0, 0, 0), restart_button, 2)
        restart_text = font.render("重新开始", True, (0, 0, 0))
        surface.blit(restart_text, (restart_button.centerx - restart_text.get_width()//2, 
                                  restart_button.centery - restart_text.get_height()//2))

# 棋盘类
class Board:
    def __init__(self, size):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.forbidden = set()
        self._init_forbidden_cells()
    
    def _init_forbidden_cells(self):
        # 随机选择四个外圈的格子作为禁止格
        outer_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 or i == 1 or i == self.size-1 or i == self.size-2 or j == 0 or j == 1 or j == self.size-1 or j == self.size-2:
                    outer_cells.append((i, j))
        self.forbidden.update(random.sample(outer_cells, 4))
    
    def draw(self, surface):
        for i in range(self.size):
            for j in range(self.size):
                rect = pygame.Rect(
                    BOARD_X + j * CELL_SIZE,
                    BOARD_Y + i * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                if (i, j) in self.forbidden:
                    pygame.draw.rect(surface, BLACK, rect)
                else:
                    pygame.draw.rect(surface, GRAY, rect)
                pygame.draw.rect(surface, WHITE, rect, 1)

# 创建游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("合成游戏")

# 游戏主循环
def main():
    clock = pygame.time.Clock()
    running = True
    
    # 初始化游戏状态和帮助弹窗
    board = Board(BOARD_SIZE)
    help_dialog = HelpDialog()
    game_over_dialog = GameOverDialog()
    game_state = GameState()
    game_state.board = board
    game_state.is_gesture_control = True
    print("已启用手势控制")
    
    while running:
        # 获取当前时间
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    # 检查是否点击了左侧材料按钮
                    for i in range(5):
                        button_y = LEFT_BUTTONS_START_Y + i * LEFT_BUTTON_SPACING
                        button_rect = pygame.Rect(LEFT_BUTTONS_X, button_y, LEFT_BUTTON_SIZE, LEFT_BUTTON_SIZE)
                        if button_rect.collidepoint(event.pos) and i < len(game_state.materials):
                            game_state.selected_material = game_state.materials[i]
                            game_state.selected_material_index = i
                            game_state.dragging = True
                            game_state.drag_pos = event.pos
                            center_i, center_j = game_state.selected_material.get_center()
                            game_state.drag_offset_x = center_j * CELL_SIZE
                            game_state.drag_offset_y = center_i * CELL_SIZE
                            game_state.message = "已选择材料"
                            game_state.message_timer = pygame.time.get_ticks()
                            break

                    # 如果结算弹窗显示，只处理弹窗按钮的点击
                    if game_over_dialog.visible:
                        dialog_x = (SCREEN_WIDTH - 400) // 2
                        dialog_y = (SCREEN_HEIGHT - 300) // 2
                        button_width = 120
                        button_height = 40
                        button_spacing = 20
                        
                        # 继续游戏按钮（仅在胜利时显示）
                        if game_over_dialog.is_victory:
                            continue_button = pygame.Rect(dialog_x + 400//2 - button_width - button_spacing//2, 
                                                        dialog_y + 300 - 80, 
                                                        button_width, button_height)
                            if continue_button.collidepoint(event.pos):
                                game_over_dialog.hide()
                                game_state.is_endless_mode = True  # 设置为无尽模式
                                game_state.message = "无尽模式已开启"
                                game_state.message_timer = pygame.time.get_ticks()
                                continue
                        
                        # 重新开始按钮
                        restart_button = pygame.Rect(dialog_x + 400//2 + button_spacing//2 if game_over_dialog.is_victory else dialog_x + 400//2 - button_width//2, 
                                                   dialog_y + 300 - 80, 
                                                   button_width, button_height)
                        if restart_button.collidepoint(event.pos):
                            game_over_dialog.hide()
                            # 重置游戏状态
                            game_state.reset()
                            # 保持手势控制开启
                            game_state.is_gesture_control = True
                            # 清空材料栏
                            game_state.materials.clear()
                            # 重新初始化棋盘
                            board = Board(BOARD_SIZE)
                            # 添加初始材料
                            for _ in range(5):
                                game_state.materials.append(Material(random.randint(1, 5)))
                            # 重置能力次数
                            game_state.yellow_ability = 0
                            game_state.purple_ability = 0
                            game_state.active_ability = None
                            game_state.message = "游戏已重新开始"
                            game_state.message_timer = pygame.time.get_ticks()
                        continue  # 如果弹窗显示，不处理其他点击事件

                    # 检查是否点击了材料
                    for i, material in enumerate(game_state.materials):
                        x = MATERIAL_START_X + i * MATERIAL_SPACING
                        y = MATERIAL_AREA_Y
                        
                        # 计算材料的实际边界
                        min_i, max_i, min_j, max_j = material.get_bounds()
                        width = (max_j - min_j + 1) * CELL_SIZE
                        height = (max_i - min_i + 1) * CELL_SIZE
                        
                        # 计算点击位置相对于材料左上角的偏移
                        click_x = event.pos[0] - x
                        click_y = event.pos[1] - y
                        
                        # 检查点击是否在材料的实际单元格内
                        if (0 <= click_x < width and 0 <= click_y < height):
                            # 将点击位置转换为材料坐标系
                            click_i = click_y // CELL_SIZE
                            click_j = click_x // CELL_SIZE
                            
                            # 检查点击位置是否在材料的实际单元格内
                            if (click_i, click_j) in material.cells:
                                # 检查是否有激活的能力
                                if game_state.active_ability == "yellow":
                                    material.joints = [True] * len(material.cells)
                                    game_state.message = "材料已变为全联结点"
                                    game_state.message_timer = pygame.time.get_ticks()
                                    game_state.active_ability = None
                                elif game_state.active_ability == "purple":
                                    material.special_place = True
                                    game_state.message = "材料已获得覆盖放置能力"
                                    game_state.message_timer = pygame.time.get_ticks()
                                    game_state.active_ability = None
                                
                            game_state.selected_material = material
                            game_state.dragging = True
                            game_state.drag_pos = event.pos
                            game_state.drag_offset_x = click_x
                            game_state.drag_offset_y = click_y
                            game_state.long_press_timer = pygame.time.get_ticks()
                            game_state.selected_material_index = i
                            break
                    
                    # 检查是否点击了获取材料按钮
                    if 415 <= event.pos[0] <= 465 and 22 <= event.pos[1] <= 48:  
                        if game_state.available_materials > 0:
                            if len(game_state.materials) < game_state.max_materials:
                                new_material = Material(random.randint(1, 5))
                                game_state.materials.append(new_material)
                                game_state.available_materials -= 1
                                game_state.message = f"获得新材料"
                            else:
                                game_state.message = "材料栏已满"
                        else:
                            game_state.message = "材料已用完"
                        game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了一键补齐按钮
                    if 470 <= event.pos[0] <= 520 and 22 <= event.pos[1] <= 48:
                        if game_state.available_materials > 0:
                            while len(game_state.materials) < game_state.max_materials and game_state.available_materials > 0:
                                game_state.materials.append(Material(random.randint(1, 5)))
                                game_state.available_materials -= 1
                            game_state.message = "材料栏已补满" if len(game_state.materials) == game_state.max_materials else f"剩余材料: {game_state.available_materials}"
                            game_state.message_timer = pygame.time.get_ticks()
                        else:
                            game_state.message = "材料已用完"
                            game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了重新开始按钮
                    if 530 <= event.pos[0] <= 630 and 22 <= event.pos[1] <= 48:
                        # 显示结算弹窗
                        game_over_dialog.show(game_state.score, game_state.connections, game_state.elimination_count)
                        # 重置游戏状态
                        game_state.reset()
                        # 清空材料栏
                        game_state.materials.clear()
                        # 重新初始化棋盘
                        board = Board(BOARD_SIZE)
                        # 添加初始材料
                        for _ in range(5):
                            game_state.materials.append(Material(random.randint(1, 5)))
                        # 重置能力次数
                        game_state.yellow_ability = 0
                        game_state.purple_ability = 0
                        game_state.active_ability = None
                        game_state.message = "游戏已重新开始"
                        game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了旋转按钮
                    if game_state.selected_material:
                        # 左旋转按钮
                        if 415 <= event.pos[0] <= 445 and 62 <= event.pos[1] <= 88:
                            game_state.selected_material.rotate(clockwise=False)
                            game_state.message = "逆时针旋转"
                            game_state.message_timer = pygame.time.get_ticks()
                        # 右旋转按钮
                        elif 450 <= event.pos[0] <= 480 and 62 <= event.pos[1] <= 88:
                            game_state.selected_material.rotate(clockwise=True)
                            game_state.message = "顺时针旋转"
                            game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了黄色能力按钮
                    if 550 <= event.pos[0] <= 600 and 62 <= event.pos[1] <= 88:
                        if game_state.yellow_ability > 0:
                            game_state.yellow_ability -= 1
                            game_state.message = "黄色能力！点击一个材料使其变为全联结点"
                            game_state.message_timer = pygame.time.get_ticks()
                            # 设置黄色能力激活状态
                            game_state.active_ability = "yellow"
                        else:
                            game_state.message = "黄色能力次数不足"
                            game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了紫色能力按钮
                    if 610 <= event.pos[0] <= 660 and 62 <= event.pos[1] <= 88:
                        if game_state.purple_ability > 0:
                            game_state.purple_ability -= 1
                            game_state.message = "紫色能力！下一个放置的材料将覆盖原有区域"
                            game_state.message_timer = pygame.time.get_ticks()
                            # 设置紫色能力激活状态
                            game_state.active_ability = "purple"
                        else:
                            game_state.message = "紫色能力次数不足"
                            game_state.message_timer = pygame.time.get_ticks()
 
                    # 检查是否点击了帮助按钮
                    elif 670 <= event.pos[0] <= 720 and 62 <= event.pos[1] <= 88:
                        help_dialog.toggle()
                        # 切换音乐播放状态
                        if game_state.music_playing:
                            pygame.mixer.music.pause()
                            game_state.music_playing = False
                        else:
                            pygame.mixer.music.unpause()
                            game_state.music_playing = True
                        game_state.message = "帮助信息已显示" if help_dialog.visible else " "
                        game_state.message_timer = pygame.time.get_ticks()
                    
                    # 点击弹窗外部关闭弹窗
                    elif help_dialog.visible:
                        dialog_rect = pygame.Rect(
                            (SCREEN_WIDTH - 400) // 2,
                            (SCREEN_HEIGHT - 400) // 2,
                            400, 400
                        )
                        if not dialog_rect.collidepoint(event.pos):
                            help_dialog.visible = False
                            # 恢复音乐播放
                            pygame.mixer.music.unpause()
                            game_state.music_playing = True
                            game_state.message = " "
                            game_state.message_timer = pygame.time.get_ticks()
                    
                    # 检查是否点击了撤销按钮
                    if 490 <= event.pos[0] <= 540 and 62 <= event.pos[1] <= 88:
                        if game_state.undo_stack:
                            # 恢复上一个状态
                            last_state = game_state.undo_stack.pop()
                            board.grid = [row[:] for row in last_state['grid']]
                            board.forbidden = set(last_state['forbidden'])
                            
                            # 清空当前材料栏
                            game_state.materials.clear()
                            
                            # 恢复材料栏中的材料
                            for mat_data in last_state['materials']:
                                if len(game_state.materials) < game_state.max_materials:
                                    # 创建新的材料对象
                                    mat = Material(mat_data['size'])
                                    # 复制材料属性
                                    mat.cells = [(int(i), int(j)) for i, j in mat_data['cells']]
                                    mat.joints = mat_data['joints']
                                    mat.color = mat_data['color']
                                    mat.special_place = mat_data['special_place']
                                    game_state.materials.append(mat)
                            
                            # 恢复分数
                            game_state.score = last_state['score']
                            game_state.message = "已撤销上一步"
                            game_state.message_timer = pygame.time.get_ticks()
                        else:
                            game_state.message = "没有可撤销的操作"
                            game_state.message_timer = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONUP:
                game_state.long_press_timer = 0
                if event.button == 1:  # 左键释放
                    if game_state.dragging:
                        game_state.dragging = False
                        # 保存当前状态到撤销栈
                        game_state.undo_stack.append({
                            'grid': [row[:] for row in board.grid],
                            'forbidden': set(board.forbidden),
                            'materials': [{
                                'size': mat.size,
                                'cells': mat.cells,
                                'joints': mat.joints,
                                'color': mat.color,
                                'special_place': mat.special_place
                            } for mat in game_state.materials],
                            'score': game_state.score,
                            'available_materials': game_state.available_materials  # 确保剩余材料数量也被保存
                        })
                        # 检查是否放置到棋盘
                        if BOARD_X <= game_state.drag_pos[0] <= BOARD_X + board.size * CELL_SIZE and \
                           BOARD_Y <= game_state.drag_pos[1] <= BOARD_Y + board.size * CELL_SIZE:
                            # 计算放置位置，考虑材料中心点偏移
                            center_i, center_j = game_state.selected_material.get_center()
                            col = (game_state.drag_pos[0] - BOARD_X) // CELL_SIZE
                            row = (game_state.drag_pos[1] - BOARD_Y) // CELL_SIZE
                            
                            # 调整放置位置，使材料中心对齐到网格
                            col = col - int(center_j)
                            row = row - int(center_i)
                            
                            # 检查放置位置是否有效
                            valid = True
                            for (i, j) in game_state.selected_material.cells:
                                r, c = row + i, col + j
                                if not (0 <= r < board.size and 0 <= c < board.size) or \
                                   (r, c) in board.forbidden or \
                                   (not game_state.selected_material.special_place and board.grid[r][c]):
                                    valid = False
                                    break
                            
                            if valid:
                                # 创建放置动画
                                game_state.animation_frames = []
                                for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                    r, c = row + i, col + j
                                    game_state.animation_frames.append({
                                        'pos': (r, c),
                                        'color': game_state.selected_material.color,
                                        'is_joint': game_state.selected_material.joints[idx]
                                    })
                                game_state.animation_active = True
                                game_state.animation_timer = pygame.time.get_ticks()
                                
                                # 放置材料到棋盘
                                for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                    r, c = row + i, col + j
                                    # 如果是紫色能力放置，先消除原有区块
                                    if game_state.selected_material.special_place and board.grid[r][c]:
                                        board.grid[r][c] = None
                                    board.grid[r][c] = {
                                        'color': game_state.selected_material.color,
                                        'is_joint': game_state.selected_material.joints[idx]
                                    }
                                
                                # 放置完成后重置特殊放置状态
                                game_state.selected_material.special_place = False
                                
                                # 计算分数
                                base_score = len(game_state.selected_material.cells)
                                joint_bonus = sum(game_state.selected_material.joints)
                                game_state.score += base_score + joint_bonus
                                
                                # 检查是否达到胜利条件
                                if game_state.score >= 100 and not game_over_dialog.visible and not game_state.is_endless_mode:
                                    game_over_dialog.show(game_state.score, game_state.connections, game_state.elimination_count, True)
                                
                                # 放置完成后调用联结数和消除检查
                                update_connections_and_eliminations(game_state, board)
                                
                                # 移除已放置的材料
                                if game_state.selected_material_index is not None:
                                    if 0 <= game_state.selected_material_index < len(game_state.materials):
                                        game_state.materials.pop(game_state.selected_material_index)
                                    game_state.selected_material = None
                                    game_state.selected_material_index = None
                                    game_state.message = "材料已放置"
                                    game_state.message_timer = pygame.time.get_ticks()
                                
                                # 重置拖放状态
                                game_state.dragging = False
                                game_state.message = "材料已放置"
                                game_state.message_timer = pygame.time.get_ticks()
                            else:
                                game_state.message = "无效的放置位置"
                                game_state.message_timer = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEMOTION:
                if game_state.dragging and game_state.selected_material:
                    # 更新拖动位置
                    game_state.drag_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # 保存游戏
                    with open('savegame.json', 'w') as f:
                        json.dump({
                            'game_state': game_state.save(),
                            'materials': [{
                                'size': mat.size,
                                'cells': mat.cells,
                                'joints': mat.joints,
                                'color': mat.color,
                                'special_place': mat.special_place
                            } for mat in game_state.materials],
                            'board': {
                                'size': board.size,
                                'grid': board.grid,
                                'forbidden': list(board.forbidden)
                            }
                        }, f)
                    game_state.message = "游戏已保存！"
                    game_state.message_timer = pygame.time.get_ticks()
                elif event.key == pygame.K_l:  # 加载游戏
                    try:
                        with open('savegame.json', 'r') as f:
                            data = json.load(f)
                            game_state.load(data['game_state'])
                            game_state.materials.clear()
                            for mat_data in data['materials']:
                                if len(game_state.materials) < game_state.max_materials:
                                    mat = Material(mat_data['size'])
                                    mat.cells = mat_data['cells']
                                    mat.joints = mat_data['joints']
                                    mat.color = mat_data['color']
                                    mat.special_place = mat_data['special_place']
                                game_state.materials.append(mat)
                            board_data = data['board']
                            board = Board(board_data['size'])
                            board.grid = board_data['grid']
                            board.forbidden = set(tuple(cell) for cell in board_data['forbidden'])
                        game_state.message = "游戏已加载！"
                        game_state.message_timer = pygame.time.get_ticks()
                    except:
                        game_state.message = "加载失败！"
                        game_state.message_timer = pygame.time.get_ticks()
                elif event.key == pygame.K_g:  # 按G键切换手势控制
                    game_state.is_gesture_control = not game_state.is_gesture_control
                    if game_state.is_gesture_control:
                        print("已启用手势控制")
                    else:
                        print("已禁用手势控制")
        
        # 填充背景
        screen.fill(WHITE)
        
        # 绘制UI背景
        pygame.draw.rect(screen, (240, 240, 240), (0, 0, SCREEN_WIDTH, UI_AREA_HEIGHT))
        
        # 绘制左侧材料按钮
        for i in range(5):
            button_y = LEFT_BUTTONS_START_Y + i * LEFT_BUTTON_SPACING
            button_rect = pygame.Rect(LEFT_BUTTONS_X, button_y, LEFT_BUTTON_SIZE, LEFT_BUTTON_SIZE)
            
            # 如果按钮对应位置有材料，使用材料颜色
            if i < len(game_state.materials):
                button_color = game_state.materials[i].color
            else:
                button_color = (200, 200, 200)  # 默认灰色
            
            # 绘制按钮
            pygame.draw.rect(screen, button_color, button_rect)
            pygame.draw.rect(screen, BLACK, button_rect, 2)  # 黑色边框
            
            # 如果按钮被选中，绘制高亮效果
            if game_state.selected_area and game_state.selected_area[0] == "left_button" and game_state.selected_area[1] == i:
                highlight_surface = pygame.Surface((LEFT_BUTTON_SIZE, LEFT_BUTTON_SIZE), pygame.SRCALPHA)
                highlight_surface.fill((255, 255, 255, 100))  # 半透明白色
                screen.blit(highlight_surface, (LEFT_BUTTONS_X, button_y))
            
            # 在按钮上显示材料编号
            number_text = font.render(str(i + 1), True, BLACK)
            number_rect = number_text.get_rect(center=button_rect.center)
            screen.blit(number_text, number_rect)
        
        # 绘制UI信息
        score_text = font.render(f"分数: {game_state.score}/100", True, BLACK)
        conn_text = font.render(f"联结数: {game_state.connections}", True, BLACK)
        mat_text = font.render(f"材料: {len(game_state.materials)}/{game_state.max_materials}", True, BLACK)
        remain_text = font.render(f"剩余: {game_state.available_materials}", True, BLACK)
        
        # 绘制基础信息
        screen.blit(score_text, (30, 20))
        screen.blit(conn_text, (30, 50))
        screen.blit(mat_text, (250, 20))
        screen.blit(remain_text, (250, 50))
        
        # 按钮文本
        get_mat_text = font.render("获取", True, BLACK)
        get_all_text = font.render("补齐", True, BLACK)
        rotate_left_text = font.render("←", True, BLACK)
        rotate_right_text = font.render("→", True, BLACK)
        undo_text = font.render("撤销", True, BLACK)
        help_button_text = font.render("帮助", True, BLACK)
        restart_text = font.render("重新开始", True, BLACK)
        
        # 材料操作按钮组
        pygame.draw.rect(screen, (180, 180, 180), (410, 20, 230, 30), border_radius=3)
        
        # 获取材料按钮
        get_mat_rect = pygame.Rect(415, 22, 50, 26)
        pygame.draw.rect(screen, (200, 200, 200), get_mat_rect, border_radius=3)
        get_mat_text_rect = get_mat_text.get_rect(center=get_mat_rect.center)
        screen.blit(get_mat_text, get_mat_text_rect)
        
        # 补齐材料按钮
        get_all_rect = pygame.Rect(470, 22, 50, 26)
        pygame.draw.rect(screen, (200, 200, 200), get_all_rect, border_radius=3)
        get_all_text_rect = get_all_text.get_rect(center=get_all_rect.center)
        screen.blit(get_all_text, get_all_text_rect)

        # 重新开始按钮
        restart_rect = pygame.Rect(530, 22, 100, 26)
        pygame.draw.rect(screen, (200, 200, 200), restart_rect, border_radius=3)
        restart_text_rect = restart_text.get_rect(center=restart_rect.center)
        screen.blit(restart_text, restart_text_rect)
        
        # 操作按钮组
        button_color = (180, 180, 180) if not game_state.selected_material else (200, 200, 200)
        pygame.draw.rect(screen, button_color, (410, 60, 320, 30), border_radius=3)
        
        # 旋转按钮组背景
        pygame.draw.rect(screen, (210, 210, 210), (415, 62, 65, 26), border_radius=3)
        
        # 旋转按钮(左)
        rotate_left_rect = pygame.Rect(415, 62, 30, 26)
        rotate_left_text_rect = rotate_left_text.get_rect(center=rotate_left_rect.center)
        screen.blit(rotate_left_text, rotate_left_text_rect)
        
        # 旋转按钮(右)
        rotate_right_rect = pygame.Rect(450, 62, 30, 26)
        rotate_right_text_rect = rotate_right_text.get_rect(center=rotate_right_rect.center)
        screen.blit(rotate_right_text, rotate_right_text_rect)
        
        # 撤销按钮
        undo_rect = pygame.Rect(490, 62, 50, 26)
        pygame.draw.rect(screen, (210, 210, 210), undo_rect, border_radius=3)
        undo_text_rect = undo_text.get_rect(center=undo_rect.center)
        screen.blit(undo_text, undo_text_rect)
        
        # 黄色能力按钮
        yellow_ability_rect = pygame.Rect(550, 62, 50, 26)
        pygame.draw.rect(screen, YELLOW, yellow_ability_rect, border_radius=3)
        yellow_ability_text = font.render("黄", True, BLACK)
        yellow_ability_text_rect = yellow_ability_text.get_rect(center=yellow_ability_rect.center)
        screen.blit(yellow_ability_text, yellow_ability_text_rect)
        
        # 紫色能力按钮
        purple_ability_rect = pygame.Rect(610, 62, 50, 26)
        pygame.draw.rect(screen, PURPLE, purple_ability_rect, border_radius=3)
        purple_ability_text = font.render("紫", True, BLACK)
        purple_ability_text_rect = purple_ability_text.get_rect(center=purple_ability_rect.center)
        screen.blit(purple_ability_text, purple_ability_text_rect)
        
        # 帮助按钮
        help_button_rect = pygame.Rect(670, 62, 50, 26)
        pygame.draw.rect(screen, (210, 210, 210), help_button_rect, border_radius=3)
        help_button_text = font.render("帮助", True, BLACK)
        help_button_text_rect = help_button_text.get_rect(center=help_button_rect.center)
        screen.blit(help_button_text, help_button_text_rect)
        
        # 能力次数显示
        pygame.draw.rect(screen, (230, 230, 230), (410, 95, 320, 25), border_radius=3)
        # 黄色能力次数
        yellow_count_text = font.render(f"黄:{game_state.yellow_ability}", True, BLACK)
        screen.blit(yellow_count_text, (485, 97))
        # 紫色能力次数
        purple_count_text = font.render(f"紫:{game_state.purple_ability}", True, BLACK)
        screen.blit(purple_count_text, (555, 97))
        
        # 绘制提示信息
        if game_state.message and pygame.time.get_ticks() - game_state.message_timer < 2000:
            msg_text = font.render(game_state.message, True, RED)
            screen.blit(msg_text, (30, 80))  
        
        # 绘制棋盘和已放置的材料
        board.draw(screen)
        
        # 绘制放置位置预览
        if game_state.dragging and game_state.selected_material:
            # 计算放置位置
            center_i, center_j = game_state.selected_material.get_center()
            col = (game_state.drag_pos[0] - BOARD_X) // CELL_SIZE
            row = (game_state.drag_pos[1] - BOARD_Y) // CELL_SIZE
            
            # 调整放置位置，使材料中心对齐到网格
            col = col - int(center_j)
            row = row - int(center_i)
            
            # 检查放置位置是否有效
            valid = True
            for (i, j) in game_state.selected_material.cells:
                r, c = row + i, col + j
                if not (0 <= r < board.size and 0 <= c < board.size) or \
                   (r, c) in board.forbidden or \
                   (not game_state.selected_material.special_place and board.grid[r][c]):
                    valid = False
                    break
            
            # 绘制预览
            if valid:
                # 创建半透明预览效果
                preview_surface = pygame.Surface((CELL_SIZE * game_state.selected_material.size, 
                                                CELL_SIZE * game_state.selected_material.size), 
                                               pygame.SRCALPHA)
                for idx, (i, j) in enumerate(game_state.selected_material.cells):
                    rect = pygame.Rect(
                        j * CELL_SIZE,
                        i * CELL_SIZE,
                        CELL_SIZE, CELL_SIZE
                    )
                    color = game_state.selected_material.color
                    if game_state.selected_material.joints[idx]:
                        color = tuple(min(c + 50, 255) for c in color)
                    # 使用更柔和的半透明效果
                    preview_surface.fill((*color, 100), rect)
                    pygame.draw.rect(preview_surface, (*color, 150), rect, 1)
                    if game_state.selected_material.joints[idx]:
                        center = (rect.x + rect.width//2, rect.y + rect.height//2)
                        radius = rect.width//4
                        pygame.draw.circle(preview_surface, (*color, 150), center, radius, 2)
                
                # 计算预览位置，使材料精确对齐到网格
                preview_x = BOARD_X + col * CELL_SIZE
                preview_y = BOARD_Y + row * CELL_SIZE
                screen.blit(preview_surface, (preview_x, preview_y))
        
        # 绘制已放置的材料
        for i in range(board.size):
            for j in range(board.size):
                if board.grid[i][j]:
                    rect = pygame.Rect(
                        BOARD_X + j * CELL_SIZE,
                        BOARD_Y + i * CELL_SIZE,
                        CELL_SIZE, CELL_SIZE
                    )
                    color = board.grid[i][j]['color']
                    if board.grid[i][j]['is_joint']:
                        color = tuple(min(c + 50, 255) for c in color)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, WHITE, rect, 1)
                    # 在联结点中心绘制反色空心圆
                    if board.grid[i][j]['is_joint']:
                        center = (rect.x + rect.width//2, rect.y + rect.height//2)
                        radius = rect.width//4
                        border_color = tuple(255 - c for c in color)
                        pygame.draw.circle(screen, border_color, center, radius, 2)
        
        # 绘制材料区
        for i, material in enumerate(game_state.materials):
            material.draw(screen, MATERIAL_START_X + i * MATERIAL_SPACING, MATERIAL_AREA_Y, CELL_SIZE)
        
        # 绘制拖动的材料
        if game_state.dragging and game_state.selected_material:
            # 使用半透明效果绘制拖动的材料
            s = pygame.Surface((CELL_SIZE * game_state.selected_material.size, 
                              CELL_SIZE * game_state.selected_material.size), 
                             pygame.SRCALPHA)
            # 使用原始材料的cells和joints信息
            for idx, (i, j) in enumerate(game_state.selected_material.cells):
                rect = pygame.Rect(
                    j * CELL_SIZE,
                    i * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                color = game_state.selected_material.color
                if game_state.selected_material.joints[idx]:  # 使用正确的联结点索引
                    color = tuple(min(c + 50, 255) for c in color)
                s.fill((*color, 200), rect)
                pygame.draw.rect(s, WHITE, rect, 1)
                if game_state.selected_material.joints[idx]:  # 使用正确的联结点索引
                    center = (rect.x + rect.width//2, rect.y + rect.height//2)
                    radius = rect.width//4
                    border_color = tuple(255 - c for c in color)
                    pygame.draw.circle(s, border_color, center, radius, 2)
            
            # 计算拖动位置，使材料中心跟随鼠标
            center_i, center_j = game_state.selected_material.get_center()
            drag_x = game_state.drag_pos[0] - center_j * CELL_SIZE
            drag_y = game_state.drag_pos[1] - center_i * CELL_SIZE
            screen.blit(s, (drag_x, drag_y))
        
        # 绘制帮助弹窗
        help_dialog.draw(screen)
        
        # 绘制结算弹窗
        game_over_dialog.draw(screen)
        
        # 在绘制摄像头画面之前，绘制手部位置指示器和吸附区域
        if game_state.is_gesture_control:
            # 绘制手部位置指示器（蓝色实心小圆圈）
            if game_state.hand_position:
                pygame.draw.circle(screen, (0, 0, 255), 
                                 (int(game_state.hand_position[0]), 
                                  int(game_state.hand_position[1])), 
                                 game_state.hand_indicator_radius)
            
            # 绘制吸附区域的高亮框
            if game_state.snap_rect:
                # 创建半透明的高亮框
                highlight_surface = pygame.Surface((game_state.snap_rect[2], 
                                                 game_state.snap_rect[3]), 
                                                pygame.SRCALPHA)
                # 绘制蓝色边框
                pygame.draw.rect(highlight_surface, (0, 0, 255, 128), 
                               (0, 0, game_state.snap_rect[2], 
                                game_state.snap_rect[3]), 2)
                # 绘制到屏幕上
                screen.blit(highlight_surface, 
                           (game_state.snap_rect[0],
                            game_state.snap_rect[1]))
        
        # 处理手势控制
        if game_state.is_gesture_control:
            # 读取摄像头画面
            ret, frame = cap.read()
            if ret:
                # 水平翻转画面
                frame = cv2.flip(frame, 1)
                
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理手势识别
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 获取手掌中心点
                        palm_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        palm_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                        
                        # 将坐标转换为屏幕坐标
                        screen_x = int(palm_x * SCREEN_WIDTH)
                        screen_y = int(palm_y * SCREEN_HEIGHT)
                        
                        # 更新手势状态
                        game_state.hand_position = (screen_x, screen_y)
                        
                        # 识别手势
                        gesture = recognize_gesture(hand_landmarks)
                        game_state.update_gesture(gesture)
                        
                        # 处理手势点击
                        if game_state.is_click() and game_state.can_click(current_time):
                            handle_gesture_click(game_state, board, screen_x, screen_y)
                        
                        # 更新选中区域
                        update_selected_area(game_state, screen_x, screen_y)
                        
                        # 检查材料栏是否为空，如果为空且还有剩余材料，自动补满
                        if len(game_state.materials) == 0 and game_state.available_materials > 0:
                            while len(game_state.materials) < game_state.max_materials and game_state.available_materials > 0:
                                game_state.materials.append(Material(random.randint(1, 5)))
                                game_state.available_materials -= 1
                            game_state.message = "材料栏已自动补充"
                            game_state.message_timer = pygame.time.get_ticks()

                        # 处理手势状态变化
                        if game_state.current_gesture == "拳头":  # 点击中
                            # 检查是否在左侧材料按钮上
                            for i in range(5):
                                button_y = LEFT_BUTTONS_START_Y + i * LEFT_BUTTON_SPACING
                                button_rect = pygame.Rect(LEFT_BUTTONS_X, button_y, LEFT_BUTTON_SIZE, LEFT_BUTTON_SIZE)
                                if button_rect.collidepoint(screen_x, screen_y) and i < len(game_state.materials):
                                    # 选择材料并开始拖放
                                    game_state.selected_material = game_state.materials[i]
                                    game_state.selected_material_index = i
                                    game_state.dragging = True
                                    game_state.drag_pos = (screen_x, screen_y)
                                    center_i, center_j = game_state.materials[i].get_center()
                                    game_state.drag_offset_x = center_j * CELL_SIZE
                                    game_state.drag_offset_y = center_i * CELL_SIZE
                                    game_state.message = "已选择材料"
                                    game_state.message_timer = pygame.time.get_ticks()
                                    break
                            # 检查是否在棋盘上
                            if (BOARD_X <= screen_x <= BOARD_X + board.size * CELL_SIZE and 
                                BOARD_Y <= screen_y <= BOARD_Y + board.size * CELL_SIZE):
                                # 如果有选中的材料，尝试放置
                                if game_state.selected_material:
                                    # 计算放置位置
                                    center_i, center_j = game_state.selected_material.get_center()
                                    col = (screen_x - BOARD_X) // CELL_SIZE
                                    row = (screen_y - BOARD_Y) // CELL_SIZE
                                    
                                    # 调整放置位置，使材料中心对齐到网格
                                    col = col - int(center_j)
                                    row = row - int(center_i)
                                    
                                    # 检查放置位置是否有效
                                    valid = True
                                    for (i, j) in game_state.selected_material.cells:
                                        r, c = row + i, col + j
                                        if not (0 <= r < board.size and 0 <= c < board.size) or \
                                           (r, c) in board.forbidden or \
                                           (not game_state.selected_material.special_place and board.grid[r][c]):
                                            valid = False
                                            break
                                    
                                    if valid:
                                        # 保存当前状态到撤销栈
                                        game_state.undo_stack.append({
                                            'grid': [row[:] for row in board.grid],
                                            'forbidden': set(board.forbidden),
                                            'materials': [{
                                                'size': mat.size,
                                                'cells': mat.cells,
                                                'joints': mat.joints,
                                                'color': mat.color,
                                                'special_place': mat.special_place
                                            } for mat in game_state.materials],
                                            'score': game_state.score,
                                            'available_materials': game_state.available_materials  # 确保剩余材料数量也被保存
                                        })
                                        # 创建放置动画
                                        game_state.animation_frames = []
                                        for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                            r, c = row + i, col + j
                                            game_state.animation_frames.append({
                                                'pos': (r, c),
                                                'color': game_state.selected_material.color,
                                                'is_joint': game_state.selected_material.joints[idx]
                                            })
                                        game_state.animation_active = True
                                        game_state.animation_timer = pygame.time.get_ticks()
                                        
                                        # 放置材料到棋盘
                                        for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                            r, c = row + i, col + j
                                            # 如果是紫色能力放置，先消除原有区块
                                            if game_state.selected_material.special_place and board.grid[r][c]:
                                                board.grid[r][c] = None
                                            board.grid[r][c] = {
                                                'color': game_state.selected_material.color,
                                                'is_joint': game_state.selected_material.joints[idx]
                                            }
                                        
                                        # 放置完成后重置特殊放置状态
                                        game_state.selected_material.special_place = False
                                        
                                        # 计算分数
                                        base_score = len(game_state.selected_material.cells)
                                        joint_bonus = sum(game_state.selected_material.joints)
                                        game_state.score += base_score + joint_bonus
                                        
                                        # 放置完成后调用联结数和消除检查
                                        update_connections_and_eliminations(game_state, board)
                                        
                                        # 移除已放置的材料
                                        if game_state.selected_material_index is not None:
                                            if 0 <= game_state.selected_material_index < len(game_state.materials):
                                                game_state.materials.pop(game_state.selected_material_index)
                                            game_state.selected_material = None
                                            game_state.selected_material_index = None
                                            game_state.message = "材料已放置"
                                            game_state.message_timer = pygame.time.get_ticks()
                                        
                                        # 重置拖放状态
                                        game_state.dragging = False
                                        game_state.message = "材料已放置"
                                        game_state.message_timer = pygame.time.get_ticks()
                                    else:
                                        game_state.message = "无效的放置位置"
                                        game_state.message_timer = pygame.time.get_ticks()
                                # 检查是否在左侧材料按钮上
                                for i in range(5):
                                    button_y = LEFT_BUTTONS_START_Y + i * LEFT_BUTTON_SPACING
                                    button_rect = pygame.Rect(LEFT_BUTTONS_X, button_y, LEFT_BUTTON_SIZE, LEFT_BUTTON_SIZE)
                                    if button_rect.collidepoint(screen_x, screen_y) and i < len(game_state.materials):
                                        # 选择材料并开始拖放
                                        game_state.selected_material = game_state.materials[i]
                                        game_state.selected_material_index = i
                                        game_state.dragging = True
                                        game_state.drag_pos = (screen_x, screen_y)
                                        center_i, center_j = game_state.materials[i].get_center()
                                        game_state.drag_offset_x = center_j * CELL_SIZE
                                        game_state.drag_offset_y = center_i * CELL_SIZE
                                        game_state.message = "已选择材料"
                                        game_state.message_timer = pygame.time.get_ticks()
                                        break
                        
                        elif game_state.current_gesture == "指向左":  # 指向左
                            # 如果正在拖放材料，向左旋转
                            if game_state.dragging and game_state.selected_material:
                                game_state.selected_material.rotate(clockwise=False)
                                game_state.message = "逆时针旋转"
                                game_state.message_timer = pygame.time.get_ticks()
                            # 如果不在拖放状态，也尝试旋转当前选中的材料
                            elif game_state.selected_material:
                                game_state.selected_material.rotate(clockwise=False)
                                game_state.message = "逆时针旋转"
                                game_state.message_timer = pygame.time.get_ticks()
                        elif game_state.current_gesture == "手掌" and game_state.previous_gesture == "拳头":  # 从点击中变为移动中
                            if game_state.dragging and game_state.selected_material:
                                # 检查是否在棋盘上
                                if (BOARD_X <= screen_x <= BOARD_X + board.size * CELL_SIZE and 
                                    BOARD_Y <= screen_y <= BOARD_Y + board.size * CELL_SIZE):
                                    # 计算放置位置
                                    center_i, center_j = game_state.selected_material.get_center()
                                    col = (screen_x - BOARD_X) // CELL_SIZE
                                    row = (screen_y - BOARD_Y) // CELL_SIZE
                                    
                                    # 调整放置位置，使材料中心对齐到网格
                                    col = col - int(center_j)
                                    row = row - int(center_i)
                                    
                                    # 检查放置位置是否有效
                                    valid = True
                                    for (i, j) in game_state.selected_material.cells:
                                        r, c = row + i, col + j
                                        if not (0 <= r < board.size and 0 <= c < board.size) or \
                                           (r, c) in board.forbidden or \
                                           (not game_state.selected_material.special_place and board.grid[r][c]):
                                            valid = False
                                            break
                                    
                                    if valid:
                                        # 保存当前状态到撤销栈
                                        game_state.undo_stack.append({
                                            'grid': [row[:] for row in board.grid],
                                            'forbidden': set(board.forbidden),
                                            'materials': [{
                                                'size': mat.size,
                                                'cells': mat.cells,
                                                'joints': mat.joints,
                                                'color': mat.color,
                                                'special_place': mat.special_place
                                            } for mat in game_state.materials],
                                            'score': game_state.score,
                                            'available_materials': game_state.available_materials  # 确保剩余材料数量也被保存
                                        })
                                        # 创建放置动画
                                        game_state.animation_frames = []
                                        for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                            r, c = row + i, col + j
                                            game_state.animation_frames.append({
                                                'pos': (r, c),
                                                'color': game_state.selected_material.color,
                                                'is_joint': game_state.selected_material.joints[idx]
                                            })
                                        game_state.animation_active = True
                                        game_state.animation_timer = pygame.time.get_ticks()
                                        
                                        # 放置材料到棋盘
                                        for idx, (i, j) in enumerate(game_state.selected_material.cells):
                                            r, c = row + i, col + j
                                            # 如果是紫色能力放置，先消除原有区块
                                            if game_state.selected_material.special_place and board.grid[r][c]:
                                                board.grid[r][c] = None
                                            board.grid[r][c] = {
                                                'color': game_state.selected_material.color,
                                                'is_joint': game_state.selected_material.joints[idx]
                                            }
                                        
                                        # 放置完成后重置特殊放置状态
                                        game_state.selected_material.special_place = False
                                        
                                        # 计算分数
                                        base_score = len(game_state.selected_material.cells)
                                        joint_bonus = sum(game_state.selected_material.joints)
                                        game_state.score += base_score + joint_bonus
                                        
                                        # 放置完成后调用联结数和消除检查
                                        update_connections_and_eliminations(game_state, board)
                                        
                                        # 移除已放置的材料
                                        if game_state.selected_material_index is not None:
                                            if 0 <= game_state.selected_material_index < len(game_state.materials):
                                                game_state.materials.pop(game_state.selected_material_index)
                                            game_state.selected_material = None
                                            game_state.selected_material_index = None
                                            game_state.message = "材料已放置"
                                            game_state.message_timer = pygame.time.get_ticks()
                                        
                                        # 重置拖放状态
                                        game_state.dragging = False
                                        game_state.message = "材料已放置"
                                        game_state.message_timer = pygame.time.get_ticks()
                                    else:
                                        game_state.message = "无效的放置位置"
                                        game_state.message_timer = pygame.time.get_ticks()
                        
                        # 绘制手部关键点
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 将OpenCV图像转换为Pygame表面
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (320, 240))
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                
                # 在屏幕右侧显示摄像头画面
                screen.blit(frame_surface, (SCREEN_WIDTH - 320, 0))
                
                # 在摄像头画面下方显示当前手势状态
                gesture_status = "等待手势"
                if game_state.current_gesture == "拳头":
                    gesture_status = "点击中"
                elif game_state.current_gesture == "手掌":
                    gesture_status = "移动中"
                elif game_state.current_gesture == "指向左":
                    gesture_status = "左旋转材料"
                elif game_state.current_gesture == "指向右":
                    gesture_status = "撤销操作"
                elif game_state.current_gesture == "指向上":
                    gesture_status = "使用黄色能力"
                elif game_state.current_gesture == "指向下":
                    gesture_status = "使用紫色能力"
                elif game_state.current_gesture == "其他":
                    gesture_status = "移动中"

                # 创建半透明的背景用于显示手势状态
                status_bg = pygame.Surface((320, 30), pygame.SRCALPHA)
                status_bg.fill((0, 0, 0, 128))  # 半透明黑色背景
                screen.blit(status_bg, (SCREEN_WIDTH - 320, 240))

                # 获取当前帧的手势识别结果
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 获取所有手指的关键点
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        
                        # 获取所有手指的MCP点
                        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                        ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                        
                        # 检查每个手指是否弯曲
                        def is_finger_bent(tip, mcp):
                            return tip.y > mcp.y - 0.05
                        
                        # 检查所有手指是否弯曲
                        thumb_bent = is_finger_bent(thumb_tip, thumb_mcp)
                        index_bent = is_finger_bent(index_tip, index_mcp)
                        middle_bent = is_finger_bent(middle_tip, middle_mcp)
                        ring_bent = is_finger_bent(ring_tip, ring_mcp)
                        pinky_bent = is_finger_bent(pinky_tip, pinky_mcp)
                        
                        # 计算伸直的手指数量
                        extended_fingers = sum([not thumb_bent, not index_bent, not middle_bent, not ring_bent, not pinky_bent])
                        gesture_text = font.render(f"{gesture_status} (伸直{extended_fingers}根手指)", True, WHITE)
                        screen.blit(gesture_text, (SCREEN_WIDTH - 320, 240))
                        break  # 只处理第一只手
                else:
                    # 如果没有检测到手，显示等待手势
                    gesture_text = font.render(f"{gesture_status} (未检测到手)", True, WHITE)
                    screen.blit(gesture_text, (SCREEN_WIDTH - 320, 240))
        
        # 在所有绘制完成后，最后绘制吸附区域的高亮框
        if game_state.is_gesture_control and game_state.snap_rect:
            # 创建半透明的高亮框
            highlight_surface = pygame.Surface((game_state.snap_rect[2], 
                                             game_state.snap_rect[3]), 
                                            pygame.SRCALPHA)
            # 绘制蓝色边框
            pygame.draw.rect(highlight_surface, (0, 0, 255, 128), 
                           (0, 0, game_state.snap_rect[2], 
                            game_state.snap_rect[3]), 2)
            # 绘制到屏幕上
            screen.blit(highlight_surface, 
                       (game_state.snap_rect[0],
                        game_state.snap_rect[1]))
        
        # 更新显示
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
