import pygame
import random
import sys

# 常量定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)

COLOR_MAP = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "purple": (128, 0, 128),
}

color_names = {
    "red": "红色",
    "yellow": "黄色",
    "green": "绿色",
    "purple": "紫色"
}

# 音乐初始化
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)  # 显式初始化
try:
    music = pygame.mixer.Sound("music.flac")  
    print("音乐加载成功")
    music_playing = True
except Exception as e:
    print(f"音乐加载失败: {e}")
    music = None
    music_playing = False

# 全局变量初始化
needs_redraw = True
info_text = None
info_start_time = 0
info_duration = 2  # 提示信息显示时长（秒）
reward_prompt = None  # 当前显示的奖励提示信息
reward_waiting = None  # 是否正在等待用户完成奖励操作
triggered_rewards = {color: set() for color in COLOR_MAP}  # 记录每种颜色已触发的奖励

# 在文件顶部定义全局变量
delta_score = {color: 0 for color in COLOR_MAP}
delta_connections = {color: 0 for color in COLOR_MAP}

# 初始化pygame
pygame.init()

# 窗口设置
WINDOW_WIDTH, WINDOW_HEIGHT = 1080, 800
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("合成材料游戏")

# 字体
try:
    font = pygame.font.SysFont("SimHei", 24)
except:
    font = pygame.font.SysFont("Arial", 24)

# 布局
BOARD_SIZE = 7
CELL_SIZE = 40
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE
BOARD_X = 100 
BOARD_Y = 90 

# 分数和联结数区域
SCORE_AREA_X = BOARD_X + BOARD_WIDTH + 100  
SCORE_AREA_Y = BOARD_Y  
SCORE_AREA_WIDTH = 500
SCORE_AREA_HEIGHT = 300

BUTTON_NAMES = ["获取", "左旋转", "右旋转", "水平翻转", "撤销", "结算"]
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 25
BUTTON_SPACING = 20
BUTTON_AREA_X = (WINDOW_WIDTH - (len(BUTTON_NAMES) * BUTTON_WIDTH + (len(BUTTON_NAMES) - 1) * BUTTON_SPACING)) // 2
BUTTON_AREA_Y = 5

MATERIALS_PER_ROW = 5
MATERIAL_WIDTH, MATERIAL_HEIGHT = 150, 150
MATERIAL_AREA_X = 20
MATERIAL_AREA_Y = BOARD_Y + BOARD_HEIGHT + 10

# 游戏状态
board = []
materials = []
score = {}
connections = {}
remaining_cost = 0
undo_stack = []
show_settlement = False
dragging = None
selected_material = None

# 在初始化时缓存按钮文本
button_texts = []
for name in BUTTON_NAMES:
    text = font.render(name, True, BLACK)
    button_texts.append(text)

# 在全局作用域中预生成透明表面
preview_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)

class GameState:
    def __init__(self):
        self.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.materials = []
        self.score = {c: 0 for c in COLOR_MAP}
        self.connections = {c: 0 for c in COLOR_MAP}
        self.remaining_cost = 15
        self.undo_stack = []
        self.delta_score = {c: 0 for c in COLOR_MAP}
        self.delta_connections = {c: 0 for c in COLOR_MAP}

    def save_state(self):
        self.undo_stack.append((
            [row.copy() for row in self.board],
            self.materials.copy(),
            self.score.copy(),
            self.connections.copy(),
            self.remaining_cost,
            {color: rewards.copy() for color, rewards in triggered_rewards.items()},
            self.delta_score.copy(),  # 新增：保存 delta_score
            self.delta_connections.copy()  # 新增：保存 delta_connections
        ))

    def undo(self):
        if not self.undo_stack:
            return
        b, m, s, c, cost, rewards, d_score, d_conn = self.undo_stack.pop()
        self.board = b
        self.materials = m
        self.score = s
        self.connections = c
        self.remaining_cost = cost
        triggered_rewards.clear()
        for color, rewards in rewards.items():
            triggered_rewards[color] = rewards.copy()
        self.delta_score = d_score  # 新增：恢复 delta_score
        self.delta_connections = d_conn  # 新增：恢复 delta_connections

def init_game():
    global board, materials, score, connections, remaining_cost, undo_stack, show_settlement, dragging, selected_material, needs_redraw, reward_waiting, info_text, info_start_time, triggered_rewards, music_playing, delta_score, delta_connections
    if music:
        music.play(loops=-1)
        music_playing = True
    else:
        print("音乐对象为 None，无法播放")
    # 重置棋盘
    board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    # 重置材料区
    materials = []
    # 重置分数和联结数
    score = {c: 0 for c in COLOR_MAP}
    connections = {c: 0 for c in COLOR_MAP}
    # 重置剩余费用
    remaining_cost = 25
    # 重置撤销栈
    undo_stack = []
    # 重置结算弹窗状态
    show_settlement = False
    # 重置拖拽状态
    dragging = None
    selected_material = None
    # 重置奖励等待状态
    reward_waiting = None
    # 重置提示信息
    info_text = None
    info_start_time = 0
    # 重置重绘状态
    needs_redraw = True
    # 重置已触发的奖励记录
    triggered_rewards = {color: set() for color in COLOR_MAP}
    # 重置 delta_score 和 delta_connections
    delta_score = {color: 0 for color in COLOR_MAP}
    delta_connections = {color: 0 for color in COLOR_MAP}

    # 初始化禁止格
    forbidden = random.sample(
        [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
         if i < 2 or i >= BOARD_SIZE-2 or j < 2 or j >= BOARD_SIZE-2], 4)
    for y, x in forbidden:
        board[y][x] = {"type": "forbidden"}

    # 初始填充材料区
    for _ in range(10):
        add_material()

def add_material(color=None, force_size=None, force_junction=None):
    global remaining_cost
    if color is None:
        if remaining_cost <= 0:
            return
        remaining_cost -= 1
        color = random.choice(list(COLOR_MAP.keys()))
    
    # 强制大小为1（单格材料）或随机
    size = 1 if force_size is not None else random.randint(1, 5)
    shape = random.choice(get_shapes(size))
    
    # 强制联结点状态或随机
    if force_junction is not None:
        junctions = {cell: force_junction for cell in shape}
    else:
        junctions = {cell: (random.random() < 0.6) for cell in shape}
    
    materials.append({"color": color, "shape": shape, "junctions": junctions})

def get_shapes(size):
    shapes = {
        1: ["0"],
        2: ["01","03"],
        3: ["012","034","045","048","042"],
        4: ["0134","0125","0158","0145","0148","0247","0248","0124","0123","0135"],
        5: ["01234","01237","13457","13478","02478","03752","03678","01247","03478","02468","01248"],
    }
    return shapes.get(size, shapes[1])

def save_state():
    undo_stack.append((
        [row.copy() for row in board],
        materials.copy(),
        score.copy(),
        connections.copy(),
        remaining_cost,
        {color: rewards.copy() for color, rewards in triggered_rewards.items()},  # 保存已触发的奖励
        delta_score.copy(),  # 新增：保存 delta_score
        delta_connections.copy()  # 新增：保存 delta_connections
    ))

def undo():
    global board, materials, score, connections, remaining_cost, needs_redraw, triggered_rewards, delta_score, delta_connections
    if not undo_stack:
        return
    b, m, s, c, cost, rewards, d_score, d_conn = undo_stack.pop()  # 新增：解包 delta_score 和 delta_connections
    board = b
    materials = m
    score = s
    connections = c
    remaining_cost = cost
    triggered_rewards = {color: rewards.copy() for color, rewards in rewards.items()}
    delta_score = d_score  # 新增：恢复 delta_score
    delta_connections = d_conn  # 新增：恢复 delta_connections
    needs_redraw = True

def can_place(mat, cx, cy):
    for cell in mat["shape"]:
        # 将形状字符转换为九宫格偏移量（0~8 对应 (-1,-1)~(1,1)）
        dx = int(cell) % 3 - 1
        dy = int(cell) // 3 - 1
        x, y = cx + dx, cy + dy
        # 检查是否在棋盘范围内
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return False
        # 检查是否为禁止格或已有材料
        if board[y][x] and board[y][x].get("type") == "forbidden":
            return False
    return True

def place(mat, cx, cy):
    save_state()
    for cell in mat["shape"]:
        dx, dy = int(cell) % 3 - 1, int(cell) // 3 - 1
        x, y = cx + dx, cy + dy
        board[y][x] = {"type": "material", "color": mat["color"], "junction": mat["junctions"][cell]}
    materials.remove(mat)
    update_scores(delta_score, delta_connections)
    eliminate()
    check_connection_rewards()  # 调用自动执行奖励逻辑
    needs_redraw = True

def update_scores(d_score=None, d_connections=None):
    global needs_redraw, delta_score, delta_connections
    # 重置分数和联结数
    for c in COLOR_MAP:
        score[c] = 0
        connections[c] = 0

    # 计算格子数（每个颜色格子+1分）
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            cell = board[i][j]
            if cell and cell.get("type") == "material":
                c = cell["color"]
                score[c] += 1  # 每个格子+1分

    # 计算联结数（仅检查右侧或下侧的相邻格子，且均为连结点）
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            cell = board[i][j]
            if cell and cell.get("type") == "material":
                c = cell["color"]
                # 仅检查右侧的相邻格子
                if j + 1 < BOARD_SIZE:
                    neighbor = board[i][j + 1]
                    if (neighbor and neighbor.get("type") == "material" and 
                        neighbor.get("color") == c and 
                        cell.get("junction") and neighbor.get("junction")):
                        connections[c] += 1  # 每对相邻连结点+1联结数
                        score[c] += 1  # 每对相邻连结点+1分
                # 仅检查下侧的相邻格子
                if i + 1 < BOARD_SIZE:
                    neighbor = board[i + 1][j]
                    if (neighbor and neighbor.get("type") == "material" and 
                        neighbor.get("color") == c and 
                        cell.get("junction") and neighbor.get("junction")):
                        connections[c] += 1  # 每对相邻连结点+1联结数
                        score[c] += 1  # 每对相邻连结点+1分

    # 如果存在差值，则加上差值
    if d_score and d_connections:
        for color in COLOR_MAP:
            score[color] -= d_score[color]
            connections[color] -= d_connections[color]

    needs_redraw = True
    check_connection_rewards()

def eliminate():
    global needs_redraw, score, connections, delta_score, delta_connections

    # 记录消除前的分数和联结数
    old_score = score.copy()
    old_connections = connections.copy()

    # 行消除（忽略禁止格）
    for row in board:
        valid_cells = [cell for cell in row if cell and cell.get("type") == "material"]
        if len(valid_cells) == sum(1 for cell in row if cell is None or cell.get("type") != "forbidden") and \
           all(cell["color"] == valid_cells[0]["color"] for cell in valid_cells):
            for j in range(BOARD_SIZE):
                if row[j] and row[j].get("type") == "material":
                    row[j] = None

            # 更新分数和联结数
            update_scores(delta_score, delta_connections)
            for color in COLOR_MAP:
                delta_score[color] += score[color] - old_score[color]
                delta_connections[color] += connections[color] - old_connections[color]
            update_scores(delta_score, delta_connections)
            needs_redraw = True

    # 列消除（忽略禁止格）
    for col_idx in range(BOARD_SIZE):
        # 提取当前列
        column = [board[row_idx][col_idx] for row_idx in range(BOARD_SIZE)]
        valid_cells = [cell for cell in column if cell and cell.get("type") == "material"]
        
        # 检查是否整列（非禁止格）为同色材料格
        if len(valid_cells) == sum(1 for cell in column if cell is None or cell.get("type") != "forbidden") and \
           all(cell["color"] == valid_cells[0]["color"] for cell in valid_cells):
            for row_idx in range(BOARD_SIZE):
                if board[row_idx][col_idx] and board[row_idx][col_idx].get("type") == "material":
                    board[row_idx][col_idx] = None

            # 更新分数和联结数
            update_scores(delta_score, delta_connections)
            for color in COLOR_MAP:
                delta_score[color] += score[color] - old_score[color]
                delta_connections[color] += connections[color] - old_connections[color]
            update_scores(delta_score, delta_connections)
            needs_redraw = True

def check_connection_rewards():
    global remaining_cost, score, connections, info_text, info_start_time, needs_redraw, triggered_rewards
    for color in COLOR_MAP:
        conn = connections[color]
        if conn >= 10 and 10 not in triggered_rewards[color]:
            remaining_cost += 3;
            info_text = font.render(f"10联结奖励：获得3cost", True, WHITE)
            info_start_time = pygame.time.get_ticks() / 1000
            needs_redraw = True
            triggered_rewards[color].add(10)
        elif conn >= 7 and 7 not in triggered_rewards[color]:
            # 7联结奖励：生成1个同色单格材料（必定为联结点）
            add_material(color=color, force_size=1, force_junction=True)
            info_text = font.render(f"7联结奖励：{color_names[color]}生成1个联结点材料", True, WHITE)
            info_start_time = pygame.time.get_ticks() / 1000
            needs_redraw = True
            triggered_rewards[color].add(7)
        elif conn >= 5 and 5 not in triggered_rewards[color]:
            # 5联结奖励：生成1个同色单格材料（必定非联结点）
            add_material(color=color, force_size=1, force_junction=False)
            info_text = font.render(f"5联结奖励：{color_names[color]}生成1个普通材料", True, WHITE)
            info_start_time = pygame.time.get_ticks() / 1000
            needs_redraw = True
            triggered_rewards[color].add(5)
        elif conn >= 3 and 3 not in triggered_rewards[color]:
            # 3联结奖励：分数+3
            score[color] += 3
            info_text = font.render(f"3联结奖励：{color_names[color]}分数+3", True, WHITE)
            info_start_time = pygame.time.get_ticks() / 1000
            needs_redraw = True
            triggered_rewards[color].add(3)

# 事件和绘制

def handle_mouse(event):
    global dragging, show_settlement, selected_material, needs_redraw, info_text, info_start_time, reward_waiting

    # 优先处理结算界面点击
    if show_settlement and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        toggle_settlement()  # 关闭结算弹窗并重置游戏
        return

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        pos = event.pos
        # 按钮逻辑
        for idx, name in enumerate(BUTTON_NAMES):
            bx = BUTTON_AREA_X + idx * (BUTTON_WIDTH + BUTTON_SPACING)
            by = BUTTON_AREA_Y
            if bx <= pos[0] <= bx + BUTTON_WIDTH and by <= pos[1] <= by + BUTTON_HEIGHT:
                if name == "获取":
                    add_material()
                    info_text = font.render("已获取新材料", True, WHITE)
                elif name == "左旋转" or name == "右旋转":
                    if selected_material:
                        rotate(selected_material, -1 if name == "左旋转" else 1)
                        info_text = font.render("材料已旋转", True, WHITE)
                elif name == "撤销":
                    undo()
                    info_text = font.render("已撤销操作", True, WHITE)
                elif name == "结算":
                    toggle_settlement()  # 打开结算界面
                elif name == "水平翻转":
                    if selected_material:
                        flip_horizontal(selected_material)
                        info_text = font.render("材料已水平翻转", True, WHITE)
                info_start_time = pygame.time.get_ticks() / 1000
                needs_redraw = True
                return

        # 材料区选取逻辑
        for mat in materials:
            idx = materials.index(mat)
            mx = MATERIAL_AREA_X + (idx % MATERIALS_PER_ROW) * MATERIAL_WIDTH
            my = MATERIAL_AREA_Y + (idx // MATERIALS_PER_ROW) * MATERIAL_HEIGHT
            if mx <= pos[0] <= mx + MATERIAL_WIDTH and my <= pos[1] <= my + MATERIAL_HEIGHT:
                selected_material = mat 
                dragging = mat
                needs_redraw = True
                return

    elif event.type == pygame.MOUSEMOTION and dragging:
        cx = (event.pos[0] - BOARD_X) // CELL_SIZE
        cy = (event.pos[1] - BOARD_Y) // CELL_SIZE
        if -1 <= cx < BOARD_SIZE + 1 and -1 <= cy < BOARD_SIZE + 1:
            dragging["pos"] = (BOARD_X + cx * CELL_SIZE + CELL_SIZE // 2, BOARD_Y + cy * CELL_SIZE + CELL_SIZE // 2)
        else:
            dragging["pos"] = event.pos
        needs_redraw = True
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
        # 放置材料逻辑
        cx = (event.pos[0] - BOARD_X) // CELL_SIZE
        cy = (event.pos[1] - BOARD_Y) // CELL_SIZE
        if can_place(dragging, cx, cy):
            place(dragging, cx, cy)
            selected_material = None
        dragging = None

def rotate(mat, d):
    """旋转材料，并触发界面重绘"""
    global needs_redraw
    if mat is None:
        return
    s = mat["shape"]

    # 将 shape 字符串转换为九宫格坐标
    cells = [int(cell) for cell in s]
    # 九宫格坐标映射：0~8 对应 (x, y)
    coords = [(cell % 3, cell // 3) for cell in cells]

    # 旋转坐标
    rotated_coords = []
    for x, y in coords:
        if d < 0:  # 左旋转（逆时针90度）
            new_x, new_y = y, 2 - x
        else:  # 右旋转（顺时针90度）
            new_x, new_y = 2 - y, x
        rotated_coords.append((new_x, new_y))

    # 将旋转后的坐标转换回 shape 字符串
    rotated_cells = [str(new_y * 3 + new_x) for new_x, new_y in rotated_coords]
    new_shape = "".join(rotated_cells)

    # 同步更新 junctions 字典的键
    old_junctions = mat["junctions"]
    new_junctions = {}
    for old_cell, new_cell in zip(s, new_shape):
        new_junctions[new_cell] = old_junctions[old_cell]

    mat["shape"] = new_shape
    mat["junctions"] = new_junctions

    needs_redraw = True

def flip_horizontal(mat):
    """水平翻转材料，并触发界面重绘"""
    global needs_redraw
    if mat is None:
        return
    s = mat["shape"]
    # 将 shape 字符串转换为九宫格坐标
    cells = [int(cell) for cell in s]
    # 九宫格坐标映射：0~8 对应 (x, y)
    coords = [(cell % 3, cell // 3) for cell in cells]

    # 水平翻转坐标
    flipped_coords = []
    for x, y in coords:
        new_x, new_y = 2 - x, y  # 水平翻转：x 坐标对称变换
        flipped_coords.append((new_x, new_y))

    # 将翻转后的坐标转换回 shape 字符串
    flipped_cells = [str(new_y * 3 + new_x) for new_x, new_y in flipped_coords]
    new_shape = "".join(flipped_cells)

    # 同步更新 junctions 字典的键
    old_junctions = mat["junctions"]
    new_junctions = {}
    for old_cell, new_cell in zip(s, new_shape):
        new_junctions[new_cell] = old_junctions[old_cell]

    mat["shape"] = new_shape
    mat["junctions"] = new_junctions
    needs_redraw = True

def toggle_settlement():
    global show_settlement, music_playing
    if show_settlement:
        show_settlement = False
        init_game()  # 关闭结算时恢复音乐
    else:
        show_settlement = True
        if music and music_playing:
            music.stop()  # 打开结算时暂停
            music_playing = False
            print("音乐已暂停")
    needs_redraw = True

def draw_progress_bar(screen, x, y, width, height, progress, max_progress, color):
    # 绘制背景
    pygame.draw.rect(screen, LIGHT_GRAY, (x, y, width, height))
    # 绘制进度
    progress_width = int(width * (progress / max_progress))
    pygame.draw.rect(screen, color, (x, y, progress_width, height))

def draw():
    global needs_redraw, info_text, info_start_time
    if not needs_redraw:
        return
    screen.fill(WHITE)
    # 使用缓存的按钮文本
    for i, text in enumerate(button_texts):
        x = BUTTON_AREA_X + i * (BUTTON_WIDTH + BUTTON_SPACING)
        screen.blit(text, text.get_rect(center=(x + BUTTON_WIDTH // 2, BUTTON_AREA_Y + BUTTON_HEIGHT // 2)))
    # 信息区（剩余费用）
    pygame.draw.rect(screen, DARK_GRAY, (0, 30, WINDOW_WIDTH, 30))
    screen.blit(font.render(f"剩余费用: {remaining_cost}", True, WHITE), (20, 30))
    # 提示信息区域
    INFO_AREA_Y = 30
    INFO_AREA_HEIGHT = 25
    
    # 检查提示信息是否超时
    current_time = pygame.time.get_ticks() / 1000  # 转换为秒
    if info_text and (current_time - info_start_time) > info_duration and not reward_waiting:
        info_text = None  # 清除提示信息

    # 显示提示信息或默认信息
    if info_text:
        screen.blit(info_text, info_text.get_rect(center=(WINDOW_WIDTH // 2, INFO_AREA_Y + INFO_AREA_HEIGHT // 2)))
    else:
        default_text = font.render(" ", True, WHITE)  # 改为黑色
        screen.blit(default_text, default_text.get_rect(center=(WINDOW_WIDTH // 2, INFO_AREA_Y + INFO_AREA_HEIGHT // 2)))
    # 棋盘
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            x, y = BOARD_X + j * CELL_SIZE, BOARD_Y + i * CELL_SIZE
            cell = board[i][j]
            color = WHITE
            brd = GRAY
            if cell:
                if cell.get("type") == "forbidden":
                    color = BLACK
                else:
                    color = COLOR_MAP[cell["color"]]
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, brd, (x, y, CELL_SIZE, CELL_SIZE), 1)
            if cell and cell.get("junction"):
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                inv = tuple(255 - c for c in color)
                pygame.draw.circle(screen, inv, (cx, cy), CELL_SIZE // 4, 1)
    # 分数和联结数区域
    pygame.draw.rect(screen, WHITE, (SCORE_AREA_X, SCORE_AREA_Y, SCORE_AREA_WIDTH, SCORE_AREA_HEIGHT))
    score_max = 25  # 统一最大值
    for idx, color in enumerate(COLOR_MAP):
        # 分数进度条
        score_progress = min(score[color], score_max)
        # 背景色（浅色）
        bg_color = (
            min(255, COLOR_MAP[color][0] + 100),
            min(255, COLOR_MAP[color][1] + 100),
            min(255, COLOR_MAP[color][2] + 100)
        )
        progress_color_color = (
            max(255, COLOR_MAP[color][0] - 100),
            max(255, COLOR_MAP[color][1] - 100),
            max(255, COLOR_MAP[color][2] - 100)
        )
        # 进度值（深色）
        progress_color = COLOR_MAP[color]
        pygame.draw.rect(screen, bg_color, (SCORE_AREA_X + 40, SCORE_AREA_Y + 40 + idx * 40, SCORE_AREA_WIDTH - 250, 20))
        pygame.draw.rect(screen, progress_color, (SCORE_AREA_X + 40, SCORE_AREA_Y + 40 + idx * 40, (SCORE_AREA_WIDTH - 250) * (score_progress / score_max), 20))
        # 分数文本（显示在进度条右侧）
        score_text = font.render(f"{color_names[color]}: {score[color]}/25", True, COLOR_MAP[color])
        screen.blit(score_text, (SCORE_AREA_X + SCORE_AREA_WIDTH - 200, SCORE_AREA_Y + 40 + idx * 40))
        # 联结数文本（显示在分数文本下方）
        conn_text = font.render(f"联结数: {connections[color]}", True, COLOR_MAP[color])
        screen.blit(conn_text, (SCORE_AREA_X + SCORE_AREA_WIDTH - 200 + 140, SCORE_AREA_Y + 40 + idx * 40))
    # 材料区及预览
    for idx, mat in enumerate(materials):
        mx = MATERIAL_AREA_X + (idx % MATERIALS_PER_ROW) * MATERIAL_WIDTH
        my = MATERIAL_AREA_Y + (idx // MATERIALS_PER_ROW) * MATERIAL_HEIGHT
        # 高亮选中的材料
        if mat == selected_material:
            pygame.draw.rect(screen, (0, 255, 0), (mx, my, MATERIAL_WIDTH, MATERIAL_HEIGHT), 3)
        else:
            pygame.draw.rect(screen, GRAY, (mx, my, MATERIAL_WIDTH, MATERIAL_HEIGHT), 1)
        # 绘制材料形状
        for cell in mat["shape"]:
            dx, dy = int(cell) % 3, int(cell) // 3  # 将字符转换为坐标
            x, y = mx + dx * CELL_SIZE, my + dy * CELL_SIZE
            pygame.draw.rect(screen, COLOR_MAP[mat["color"]], (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
            if mat["junctions"][cell]:
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                inv = tuple(255 - c for c in COLOR_MAP[mat["color"]])
                pygame.draw.circle(screen, inv, (cx, cy), CELL_SIZE // 4, 1)
    if dragging:
        px, py = dragging.get("pos", pygame.mouse.get_pos())
        for cell in dragging["shape"]:
            dx, dy = int(cell) % 3, int(cell) // 3
            x, y = px + dx * CELL_SIZE - 60, py + dy * CELL_SIZE - 60
            preview_surface.fill((*COLOR_MAP[dragging["color"]], 128))  # 半透明填充
            screen.blit(preview_surface, (x, y))
            pygame.draw.rect(screen, (0, 0, 255), (x, y, CELL_SIZE, CELL_SIZE), 1)
            # 绘制联结点（如果存在）
            if dragging["junctions"].get(cell, False):
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                inv = tuple(255 - c for c in COLOR_MAP[dragging["color"]])
                pygame.draw.circle(screen, inv, (cx, cy), CELL_SIZE // 4, 1)
    # 结算弹窗
    if show_settlement:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        w, h = 700, 300
        x = (WINDOW_WIDTH - w) // 2
        y = (WINDOW_HEIGHT - h) // 2
        pygame.draw.rect(screen, WHITE, (x, y, w, h))
        screen.blit(font.render("游戏结算", True, BLACK), (x + w//2 -50, y + 20))
        yy = y + 70
        score_max = 25  # 统一最大值
        for c in COLOR_MAP:
            # 进度条背景（浅色）
            bg_color = (
                min(255, COLOR_MAP[c][0] + 100),
                min(255, COLOR_MAP[c][1] + 100),
                min(255, COLOR_MAP[c][2] + 100)
            )
            # 进度值（深色）
            progress_color = COLOR_MAP[c]
            pygame.draw.rect(screen, bg_color, (x + 50, yy, 270, 20))
            pygame.draw.rect(screen, progress_color, (x + 50, yy, 270 * (min(score[c], score_max) / score_max), 20))
            # 分数文本
            score_text = font.render(f"{color_names[c]}: {score[c]}/25", True, COLOR_MAP[c])
            conn_text = font.render(f"联结数: {connections[c]}", True, COLOR_MAP[c])
            screen.blit(score_text, (x + 340, yy))
            screen.blit(conn_text, (x + 480, yy))
            yy += 40
        screen.blit(font.render("点击任意位置关闭", True, BLACK), (x + w//2 - 100, y + h - 50))
    pygame.display.flip()
    needs_redraw = False

# 主循环
init_game()
clock = pygame.time.Clock()
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            if music:
                music.stop()
            pygame.quit()
            sys.exit()
        if reward_waiting and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            reward_waiting = None
            needs_redraw = True
        elif not reward_waiting:
            handle_mouse(e)
    draw()
    clock.tick(60)
