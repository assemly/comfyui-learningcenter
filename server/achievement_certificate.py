import os
import json
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import tempfile
import traceback
import random

# 尝试导入torch，如果失败则提供警告
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("[成就系统] 警告: 未能导入PyTorch，将使用NumPy数组作为输出")
    TORCH_AVAILABLE = False

# 获取当前目录
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 成就级别及其所需的完成章节数
ACHIEVEMENT_LEVELS = {
    "初学者": 1,      # 完成1个章节
    "学徒": 5,        # 完成5个章节
    "实践者": 10,     # 完成10个章节
    "专家": 20,       # 完成20个章节
    "大师": 30,       # 完成30个章节
    "宗师": 50        # 完成50个章节
}

# 隐藏成就定义
HIDDEN_ACHIEVEMENTS = {
    "opai_fan": {
        "name": "Opai粉丝",
        "description": "尝试过所有Opai吉祥物风格的忠实粉丝",
        "unlocked": False
    }
}

class AchievementCertificate:
    """生成ComfyUI学习成就证书的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入类型"""
        return {
            "required": {
                "user_name": ("STRING", {"default": "请输入你的姓名", "multiline": False}),
                "certificate_style": (["标准", "卡通", "复古", "未来科技", "手绘风"], {"default": "标准"}),
                "achievement_title": ("STRING", {"default": "ComfyUI学习成就", "multiline": False}),
                "font_size": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1}),
                "text_color": (["黑色", "白色", "红色", "蓝色", "绿色", "金色", "紫色", "橙色"], {"default": "黑色"}),
                "bg_color": (["白色", "透明", "金色", "淡蓝色", "淡绿色", "渐变蓝", "渐变粉"], {"default": "白色"}),
                "show_mascot": (["是", "否"], {"default": "是", "description": "显示Opai吉祥物"}),
                "mascot_style": (["正常", "欢呼", "鼓掌", "学术帽", "随机"], {"default": "正常", "description": "Opai吉祥物的风格"}),
                "border_style": (["无", "简约", "金色华丽", "线条", "气泡", "像素风"], {"default": "简约"}),
            },
            "hidden": {
                "override_level": ("BOOLEAN", {"default": False}),
                "manual_level": (["初学者", "学徒", "实践者", "专家", "大师", "宗师"], {"default": "初学者"})
            },
            "optional": {
                "custom_image": ("IMAGE",),
                "fun_quote": ("STRING", {"default": "", "multiline": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "generate_certificate"
    CATEGORY = "学习中心/成就"
    
    def generate_certificate(self, user_name, certificate_style, achievement_title, font_size, text_color, bg_color, show_mascot, mascot_style, border_style, custom_image=None, fun_quote="", override_level=False, manual_level="初学者"):
        """生成成就证书图像"""
        try:
            # 从progress.json获取完成的章节数
            completed_count = self.get_completed_chapters_count()
            print(f"[成就系统] 从进度文件读取到已完成章节数: {completed_count}")
            
            # 根据完成的章节数确定成就级别，除非在高级选项中被覆盖
            if override_level:
                level = manual_level
                print(f"[成就系统] 管理员手动指定成就级别: {level}")
            else:
                level = self.get_achievement_level(completed_count)
                print(f"[成就系统] 根据完成章节数({completed_count})自动确定成就级别: {level}")
            
            # 检查隐藏成就
            hidden_achievements = self.check_hidden_achievements(mascot_style)
            
            # 根据输入的级别和完成章节数生成证书
            certificate_image = self.create_certificate(
                completed_count, 
                level, 
                user_name, 
                certificate_style, 
                achievement_title,
                font_size, 
                text_color, 
                bg_color, 
                show_mascot, 
                mascot_style,
                border_style,
                custom_image,
                fun_quote=fun_quote,
                hidden_achievements=hidden_achievements
            )
            
            # 可选：保存到临时文件
            temp_dir = tempfile.gettempdir()
            certificate_filename = f"comfyui_achievement_{int(time.time())}.png"
            certificate_path = os.path.join(temp_dir, certificate_filename)
            certificate_image.save(certificate_path)
            print(f"[成就系统] 证书已保存至: {certificate_path}")
            
            # 确保图像是RGBA格式
            if certificate_image.mode != 'RGBA':
                certificate_image = certificate_image.convert('RGBA')
            
            # 将PIL图像转换为NumPy数组
            np_image = np.array(certificate_image).astype(np.float32) / 255.0
            
            # 检查是否可以使用PyTorch
            if TORCH_AVAILABLE:
                try:
                    # 转换为PyTorch张量并确保格式正确
                    # 格式为[batch, height, width, channel]
                    torch_image = torch.from_numpy(np_image)
                    if len(torch_image.shape) == 3:  # [height, width, channel]
                        torch_image = torch_image.unsqueeze(0)  # 添加batch维度
                    
                    return (torch_image,)
                except Exception as e:
                    print(f"[成就系统] 警告: 转换为PyTorch张量时出错: {e}，将使用NumPy数组")
            
            # 如果PyTorch不可用或转换失败，返回NumPy数组
            # 返回格式为[batch, height, width, channel]的图像
            np_image = np_image[None, :, :, :]
            
            return (np_image,)
            
        except Exception as e:
            print(f"[成就系统] 生成证书时出错: {e}")
            traceback.print_exc()
            # 返回一个小的错误图像
            error_img = np.zeros((1, 256, 256, 3), dtype=np.float32)
            
            if TORCH_AVAILABLE:
                return (torch.from_numpy(error_img),)
            return (error_img,)
    
    def get_completed_chapters_count(self):
        """获取用户完成的章节数"""
        progress_file = os.path.join(current_dir, "user_progress", "progress.json")
        if not os.path.exists(progress_file):
            return 0
        
        try:
            with open(progress_file, "r", encoding="utf-8-sig") as f:
                progress_data = json.load(f)
                completed_chapters = progress_data.get("completed_chapters", {})
                return len(completed_chapters)
        except Exception as e:
            print(f"[成就系统] 无法读取用户进度: {e}")
            return 0
    
    def get_achievement_level(self, completed_count):
        """根据完成的章节数确定成就级别"""
        level = "初学者"  # 默认级别
        
        for level_name, required_count in sorted(ACHIEVEMENT_LEVELS.items(), key=lambda x: x[1], reverse=True):
            if completed_count >= required_count:
                level = level_name
                break
                
        return level
    
    def get_next_achievement_level(self, completed_count):
        """获取下一个成就级别和需要完成的章节数"""
        current_level = self.get_achievement_level(completed_count)
        
        # 获取所有级别，按照所需章节数升序排序
        sorted_levels = sorted(ACHIEVEMENT_LEVELS.items(), key=lambda x: x[1])
        
        next_level = None
        chapters_needed = 0
        
        for level_name, required_count in sorted_levels:
            if required_count > completed_count:
                next_level = level_name
                chapters_needed = required_count - completed_count
                break
        
        if next_level is None:
            # 如果已经是最高级别
            return None, 0
        
        return next_level, chapters_needed
    
    def check_hidden_achievements(self, current_mascot_style):
        """检查隐藏成就是否达成"""
        # 获取用户的Opai风格使用历史
        hidden_achievements = dict(HIDDEN_ACHIEVEMENTS)
        try:
            # 读取隐藏成就记录文件
            achievements_file = os.path.join(current_dir, "user_progress", "hidden_achievements.json")
            if not os.path.exists(achievements_file):
                # 初始化文件
                os.makedirs(os.path.dirname(achievements_file), exist_ok=True)
                with open(achievements_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "opai_styles_used": [],
                        "unlocked_achievements": []
                    }, f, ensure_ascii=False, indent=2)
            
            # 读取当前成就状态
            with open(achievements_file, "r", encoding="utf-8") as f:
                achievements_data = json.load(f)
                
            # 获取已使用的Opai风格
            opai_styles_used = achievements_data.get("opai_styles_used", [])
            unlocked_achievements = achievements_data.get("unlocked_achievements", [])
            
            # 添加当前使用的风格
            if current_mascot_style not in opai_styles_used:
                opai_styles_used.append(current_mascot_style)
                # 更新文件
                achievements_data["opai_styles_used"] = opai_styles_used
                with open(achievements_file, "w", encoding="utf-8") as f:
                    json.dump(achievements_data, f, ensure_ascii=False, indent=2)
                print(f"[成就系统] 记录新的Opai风格使用: {current_mascot_style}，当前已使用: {opai_styles_used}")
            
            # 检查是否解锁"Opai粉丝"成就
            all_styles = ["正常", "欢呼", "鼓掌", "学术帽", "随机"]
            is_opai_fan = all(style in opai_styles_used for style in all_styles)
            
            if is_opai_fan and "opai_fan" not in unlocked_achievements:
                # 解锁成就
                unlocked_achievements.append("opai_fan")
                achievements_data["unlocked_achievements"] = unlocked_achievements
                with open(achievements_file, "w", encoding="utf-8") as f:
                    json.dump(achievements_data, f, ensure_ascii=False, indent=2)
                print("[成就系统] 解锁隐藏成就: Opai粉丝！已尝试所有Opai风格")
                hidden_achievements["opai_fan"]["unlocked"] = True
            elif "opai_fan" in unlocked_achievements:
                hidden_achievements["opai_fan"]["unlocked"] = True
                
        except Exception as e:
            print(f"[成就系统] 检查隐藏成就出错: {e}")
            
        return hidden_achievements
    
    def create_certificate(self, completed_count, level, user_name, certificate_style, achievement_title, font_size, text_color, bg_color, show_mascot, mascot_style, border_style, custom_image=None, fun_quote="", hidden_achievements=None):
        """创建证书图像"""
        width, height = 1024, 768
        
        # 设置背景颜色
        bg_color_map = {
            "透明": (0, 0, 0, 0),
            "白色": (255, 255, 255, 255),
            "金色": (253, 240, 213, 255),
            "淡蓝色": (235, 245, 255, 255),
            "淡绿色": (240, 255, 240, 255)
        }
        
        # 创建背景
        if bg_color == "透明":
            certificate = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        elif bg_color == "渐变蓝":
            # 创建渐变背景
            certificate = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for y in range(height):
                r = int(235 - 30 * y / height)
                g = int(245 - 20 * y / height)
                b = int(255)
                for x in range(width):
                    certificate.putpixel((x, y), (r, g, b, 255))
        elif bg_color == "渐变粉":
            # 创建渐变背景
            certificate = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for y in range(height):
                r = int(255 - 10 * y / height)
                g = int(220 - 50 * y / height)
                b = int(240 - 40 * y / height)
                for x in range(width):
                    certificate.putpixel((x, y), (r, g, b, 255))
        else:
            certificate = Image.new("RGBA", (width, height), bg_color_map[bg_color])
        
        # 根据证书风格添加额外的背景元素
        if certificate_style == "卡通":
            # 添加卡通气泡背景
            try:
                for i in range(20):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    size = np.random.randint(20, 100)
                    color = (
                        np.random.randint(200, 255),
                        np.random.randint(200, 255),
                        np.random.randint(200, 255),
                        np.random.randint(30, 100)
                    )
                    ellipse_shape = [(x-size//2, y-size//2), (x+size//2, y+size//2)]
                    draw = ImageDraw.Draw(certificate)
                    draw.ellipse(ellipse_shape, fill=color)
            except Exception as e:
                print(f"[成就系统] 创建卡通背景元素时出错: {e}")
        
        elif certificate_style == "复古":
            # 添加复古纹理
            try:
                # 创建噪点纹理
                for i in range(width*height//100):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    size = np.random.randint(1, 3)
                    brightness = np.random.randint(0, 50)
                    color = (brightness, brightness, brightness, 50)
                    draw = ImageDraw.Draw(certificate)
                    draw.rectangle([(x, y), (x+size, y+size)], fill=color)
            except Exception as e:
                print(f"[成就系统] 创建复古纹理时出错: {e}")
        
        elif certificate_style == "未来科技":
            # 添加科技感线条
            try:
                draw = ImageDraw.Draw(certificate)
                for i in range(20):
                    x1 = np.random.randint(0, width//4)
                    y1 = np.random.randint(0, height)
                    x2 = np.random.randint(3*width//4, width)
                    y2 = np.random.randint(0, height)
                    color = (0, 150, 255, np.random.randint(10, 30))
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
            except Exception as e:
                print(f"[成就系统] 创建科技感线条时出错: {e}")
        
        elif certificate_style == "手绘风":
            # 添加手绘风效果
            try:
                # 在边缘添加一些随机的短线
                draw = ImageDraw.Draw(certificate)
                for i in range(100):
                    edge = np.random.randint(0, 4)  # 0=上, 1=右, 2=下, 3=左
                    if edge == 0:
                        x1 = np.random.randint(0, width)
                        y1 = np.random.randint(0, 50)
                    elif edge == 1:
                        x1 = np.random.randint(width-50, width)
                        y1 = np.random.randint(0, height)
                    elif edge == 2:
                        x1 = np.random.randint(0, width)
                        y1 = np.random.randint(height-50, height)
                    else:
                        x1 = np.random.randint(0, 50)
                        y1 = np.random.randint(0, height)
                    
                    length = np.random.randint(5, 15)
                    angle = np.random.random() * np.pi * 2
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))
                    color = (100, 100, 100, 30)
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
            except Exception as e:
                print(f"[成就系统] 创建手绘风效果时出错: {e}")
        
        # 重新获取绘图对象（因为在某些样式处理中可能已经创建）
        draw = ImageDraw.Draw(certificate)
        
        # 设置文字颜色
        text_color_map = {
            "白色": (255, 255, 255, 255),
            "黑色": (0, 0, 0, 255),
            "红色": (200, 30, 30, 255),
            "蓝色": (30, 30, 200, 255),
            "绿色": (30, 150, 50, 255),
            "金色": (212, 175, 55, 255),
            "紫色": (128, 0, 128, 255),
            "橙色": (255, 140, 0, 255)
        }
        
        # 添加边框
        if border_style != "无":
            border_width = 10
            if border_style == "金色华丽":
                # 绘制金色华丽边框
                for i in range(border_width):
                    # 渐变金色
                    color = (212, 175, 55, 255 - i * 15)
                    draw.rectangle(
                        [i, i, width - i - 1, height - i - 1],
                        outline=color
                    )
            elif border_style == "简约":
                # 简约边框
                draw.rectangle(
                    [border_width, border_width, width - border_width - 1, height - border_width - 1],
                    outline=text_color_map[text_color],
                    width=3
                )
            elif border_style == "线条":
                # 线条边框
                for i in range(0, width, 20):
                    draw.line([(i, 0), (i, 10)], fill=text_color_map[text_color], width=2)
                    draw.line([(i, height-10), (i, height)], fill=text_color_map[text_color], width=2)
                for i in range(0, height, 20):
                    draw.line([(0, i), (10, i)], fill=text_color_map[text_color], width=2)
                    draw.line([(width-10, i), (width, i)], fill=text_color_map[text_color], width=2)
            elif border_style == "气泡":
                # 气泡边框
                for i in range(0, width, 40):
                    circle_radius = 10
                    draw.ellipse([(i-circle_radius, 0-circle_radius), (i+circle_radius, 0+circle_radius)], 
                                fill=text_color_map[text_color])
                    draw.ellipse([(i-circle_radius, height-circle_radius), (i+circle_radius, height+circle_radius)], 
                                fill=text_color_map[text_color])
                for i in range(0, height, 40):
                    circle_radius = 10
                    draw.ellipse([(0-circle_radius, i-circle_radius), (0+circle_radius, i+circle_radius)], 
                                fill=text_color_map[text_color])
                    draw.ellipse([(width-circle_radius, i-circle_radius), (width+circle_radius, i+circle_radius)], 
                                fill=text_color_map[text_color])
            elif border_style == "像素风":
                # 像素风边框
                pixel_size = 15
                for i in range(0, width, pixel_size):
                    if np.random.random() > 0.3:  # 70% 的像素会被绘制
                        draw.rectangle([(i, 0), (i+pixel_size-1, pixel_size-1)], 
                                    fill=text_color_map[text_color])
                        draw.rectangle([(i, height-pixel_size), (i+pixel_size-1, height-1)], 
                                    fill=text_color_map[text_color])
                for i in range(0, height, pixel_size):
                    if np.random.random() > 0.3:  # 70% 的像素会被绘制
                        draw.rectangle([(0, i), (pixel_size-1, i+pixel_size-1)], 
                                    fill=text_color_map[text_color])
                        draw.rectangle([(width-pixel_size, i), (width-1, i+pixel_size-1)], 
                                    fill=text_color_map[text_color])
        
        # 尝试加载字体
        try:
            # 尝试加载Windows系统的字体
            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/simhei.ttf"  # 尝试黑体
            
            title_font = ImageFont.truetype(font_path, font_size + 12)
            main_font = ImageFont.truetype(font_path, font_size)
            sub_font = ImageFont.truetype(font_path, font_size - 10)
            name_font = ImageFont.truetype(font_path, font_size + 6)  # 用户名字体
        except Exception as e:
            print(f"[成就系统] 无法加载字体: {e}，使用默认字体")
            # 使用默认字体
            title_font = ImageFont.load_default()
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
            name_font = ImageFont.load_default()
        
        # 添加标题
        title = f"{achievement_title}证书"
        title_width = draw.textlength(title, font=title_font)
        draw.text(
            ((width - title_width) // 2, 60),
            title,
            font=title_font,
            fill=text_color_map[text_color]
        )
        
        # 添加用户名称
        user_text = f"授予: {user_name}"
        user_width = draw.textlength(user_text, font=name_font)
        
        # 为用户名字添加背景高亮效果
        highlight_padding = 15  # 增加内边距
        highlight_height = font_size + 30  # 增加高度
        highlight_width = user_width + highlight_padding * 2
        
        # 根据证书风格选择不同的高亮效果
        if certificate_style == "卡通":
            # 卡通风格使用圆形高亮
            highlight_color = (255, 255, 200, 150)  # 淡黄色半透明
            draw.ellipse(
                [((width - highlight_width) // 2, 170), 
                 ((width + highlight_width) // 2, 170 + highlight_height)],
                fill=highlight_color
            )
        elif certificate_style == "复古":
            # 复古风格使用矩形高亮
            highlight_color = (240, 240, 220, 150)  # 米色半透明
            draw.rectangle(
                [((width - highlight_width) // 2, 170), 
                 ((width + highlight_width) // 2, 170 + highlight_height)],
                fill=highlight_color
            )
        elif certificate_style == "未来科技":
            # 科技风格使用六边形高亮
            highlight_color = (200, 230, 255, 150)  # 淡蓝色半透明
            # 绘制六边形
            hex_points = []
            center_x = width // 2
            center_y = 170 + highlight_height // 2
            for i in range(6):
                angle = i * 60 - 30  # 从-30度开始，每60度一个点
                rad = np.radians(angle)
                x = center_x + (highlight_width // 2) * np.cos(rad)
                y = center_y + (highlight_height // 2) * np.sin(rad)
                hex_points.append((x, y))
            draw.polygon(hex_points, fill=highlight_color)
        elif certificate_style == "手绘风":
            # 手绘风格使用不规则形状高亮
            highlight_color = (255, 250, 240, 150)  # 淡米色半透明
            # 绘制不规则形状
            points = [
                ((width - highlight_width) // 2, 170),
                ((width - highlight_width) // 2 + 20, 170 - 5),
                ((width + highlight_width) // 2 - 20, 170 - 5),
                ((width + highlight_width) // 2, 170),
                ((width + highlight_width) // 2, 170 + highlight_height),
                ((width + highlight_width) // 2 - 20, 170 + highlight_height + 5),
                ((width - highlight_width) // 2 + 20, 170 + highlight_height + 5),
            ]
            draw.polygon(points, fill=highlight_color)
        else:
            # 标准风格使用简单矩形高亮
            highlight_color = (240, 240, 240, 150)  # 淡灰色半透明
            draw.rectangle(
                [((width - highlight_width) // 2, 170), 
                 ((width + highlight_width) // 2, 170 + highlight_height)],
                fill=highlight_color
            )
        
        # 绘制用户名文本
        draw.text(
            ((width - user_width) // 2, 180),
            user_text,
            font=name_font,
            fill=text_color_map[text_color]
        )
        
        # 添加成就级别
        level_text = f"成就等级: {level}"
        level_width = draw.textlength(level_text, font=main_font)
        draw.text(
            ((width - level_width) // 2, 240),
            level_text,
            font=main_font,
            fill=text_color_map[text_color]
        )
        
        # 添加完成章节数
        chapters_text = f"已完成章节: {completed_count}"
        chapters_width = draw.textlength(chapters_text, font=main_font)
        draw.text(
            ((width - chapters_width) // 2, 290),
            chapters_text,
            font=main_font,
            fill=text_color_map[text_color]
        )
        
        # 生成有趣的成就描述
        achievement_descriptions = {
            "初学者": ["踏出了探索ComfyUI的第一步", "开启了AI绘画的奇妙旅程", "已经迈入了AI艺术的大门"],
            "学徒": ["在ComfyUI的道路上稳步前进", "掌握了基础知识", "成为了AI绘画的学徒"],
            "实践者": ["已能熟练使用ComfyUI", "掌握了多种工作流技巧", "成为了ComfyUI的实践达人"],
            "专家": ["精通ComfyUI的各种用法", "可以解决复杂的AI绘画问题", "已成为值得尊敬的专家"],
            "大师": ["对ComfyUI了如指掌", "创造性地使用各种工作流", "真正的ComfyUI大师"],
            "宗师": ["登峰造极的ComfyUI专家", "已达到罕见的技术高度", "站在AI绘画艺术的顶峰"]
        }
        
        description = np.random.choice(achievement_descriptions.get(level, ["很棒的成就"]))
        description_text = f"「{description}」"
        description_width = draw.textlength(description_text, font=main_font)
        draw.text(
            ((width - description_width) // 2, 340),
            description_text,
            font=main_font,
            fill=text_color_map[text_color]
        )
        
        # 设置初始y位置
        y_position = 390
        
        # 添加自定义引言（如果有）
        if fun_quote:
            text_lines = fun_quote.strip().split("\n")
            for line in text_lines:
                line_width = draw.textlength(line, font=main_font)
                draw.text(
                    ((width - line_width) // 2, y_position),
                    line,
                    font=main_font,
                    fill=text_color_map[text_color]
                )
                y_position += font_size + 10
        else:
            # 添加随机励志语录
            motivational_quotes = [
                "创造力是最好的奖励！",
                "每一个工作流都是新的开始",
                "持续学习，无限可能",
                "AI艺术的世界等你探索",
                "成为ComfyUI大师的路上不止一个里程碑",
                "你的创意，AI的翅膀",
                "每一章都是新的冒险",
                "学习的道路没有尽头",
                "与AI共创美好未来",
                "今天的学习，明天的杰作"
            ]
            
            random_quote = np.random.choice(motivational_quotes)
            quote_width = draw.textlength(random_quote, font=main_font)
            draw.text(
                ((width - quote_width) // 2, y_position),
                random_quote,
                font=main_font,
                fill=text_color_map[text_color]
            )
        
        # 添加日期
        current_date = time.strftime("%Y年%m月%d日")
        date_text = f"颁发日期: {current_date}"
        date_width = draw.textlength(date_text, font=sub_font)
        draw.text(
            ((width - date_width) // 2, height - 120),
            date_text,
            font=sub_font,
            fill=text_color_map[text_color]
        )
        
        # 添加吉祥物图片
        if show_mascot == "是":
            mascot_path = os.path.join(current_dir, "resources", "opai.png")
            
            # 检查是否解锁了Opai粉丝成就
            if hidden_achievements and hidden_achievements.get("opai_fan", {}).get("unlocked", False):
                # 使用特殊的Opai图像（彩虹效果）
                print("[成就系统] 使用特殊的Opai粉丝版本图像")
                try:
                    mascot = None
                    if os.path.exists(mascot_path):
                        mascot = Image.open(mascot_path).convert("RGBA")
                        # 调整大小
                        mascot_width = 180
                        mascot_height = int(mascot.height * mascot_width / mascot.width)
                        mascot = mascot.resize((mascot_width, mascot_height), Image.LANCZOS)
                        
                        # 为Opai添加彩虹光环效果
                        # 创建一个新的透明图层
                        rainbow_layer = Image.new("RGBA", (mascot_width + 40, mascot_height + 40), (0, 0, 0, 0))
                        draw_rainbow = ImageDraw.Draw(rainbow_layer)
                        
                        # 添加彩虹环
                        rainbow_colors = [
                            (255, 0, 0, 150),    # 红色
                            (255, 165, 0, 150),  # 橙色
                            (255, 255, 0, 150),  # 黄色
                            (0, 255, 0, 150),    # 绿色
                            (0, 0, 255, 150),    # 蓝色
                            (75, 0, 130, 150),   # 靛色
                            (238, 130, 238, 150) # 紫色
                        ]
                        
                        # 在Opai周围绘制彩虹光环
                        center_x, center_y = mascot_width//2 + 20, mascot_height//2 + 20
                        for i, color in enumerate(rainbow_colors):
                            radius = mascot_width//2 + 20 - i*3
                            draw_rainbow.ellipse(
                                [(center_x - radius, center_y - radius), 
                                 (center_x + radius, center_y + radius)],
                                outline=color,
                                width=5
                            )
                        
                        # 在彩虹层中央粘贴Opai图像
                        rainbow_layer.paste(mascot, (20, 20), mascot)
                        
                        # 添加文本"Opai粉丝"
                        try:
                            # 在Opai上方添加文本
                            font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
                            if not os.path.exists(font_path):
                                font_path = "C:/Windows/Fonts/simhei.ttf"
                            fan_font = ImageFont.truetype(font_path, 24)
                            fan_text = "★ Opai粉丝 ★"
                            draw_rainbow = ImageDraw.Draw(rainbow_layer)
                            text_width = draw_rainbow.textlength(fan_text, font=fan_font)
                            draw_rainbow.text(
                                ((rainbow_layer.width - text_width) // 2, 0),
                                fan_text,
                                font=fan_font,
                                fill=(255, 215, 0, 255)  # 金色
                            )
                        except Exception as e:
                            print(f"[成就系统] 添加Opai粉丝文本失败: {e}")
                        
                        # 使用彩虹效果层替代原始mascot
                        mascot = rainbow_layer
                        
                        # 放置特殊版本的Opai
                        certificate.paste(mascot, (width - mascot.width - 30, height - mascot.height - 30), mascot)
                        print("[成就系统] 成功添加Opai粉丝特别版图像")
                        
                        # 添加隐藏成就解锁提示
                        try:
                            achievement_font = ImageFont.truetype(font_path, 20)
                            achievement_text = "★ 解锁隐藏成就：Opai粉丝 ★"
                            achievement_width = draw.textlength(achievement_text, font=achievement_font)
                            # 在证书底部添加成就解锁提示
                            draw.text(
                                ((width - achievement_width) // 2, height - 80),
                                achievement_text,
                                font=achievement_font,
                                fill=(255, 215, 0, 255)  # 金色
                            )
                        except Exception as e:
                            print(f"[成就系统] 添加隐藏成就文本失败: {e}")
                    
                except Exception as e:
                    print(f"[成就系统] 创建Opai粉丝特别版图像失败: {e}")
                    # 回退到普通的Opai图像
                    if os.path.exists(mascot_path):
                        mascot = Image.open(mascot_path).convert("RGBA")
                        # 调整大小并放置在右下角
                        mascot_width = 180
                        mascot_height = int(mascot.height * mascot_width / mascot.width)
                        mascot = mascot.resize((mascot_width, mascot_height), Image.LANCZOS)
                        certificate.paste(mascot, (width - mascot_width - 50, height - mascot_height - 50), mascot)
            else:
                # 使用普通的Opai图像
                if os.path.exists(mascot_path):
                    try:
                        print(f"[成就系统] 加载Opai吉祥物图像: {mascot_path}")
                        mascot = Image.open(mascot_path).convert("RGBA")
                        # 调整大小并放置在右下角
                        mascot_width = 180
                        mascot_height = int(mascot.height * mascot_width / mascot.width)
                        mascot = mascot.resize((mascot_width, mascot_height), Image.LANCZOS)
                        
                        # 根据选择的吉祥物风格进行处理
                        if mascot_style == "欢呼":
                            # 稍微旋转吉祥物图像
                            print("[成就系统] 应用'欢呼'风格到Opai")
                            mascot = mascot.rotate(15, resample=Image.BICUBIC, expand=True)
                        elif mascot_style == "鼓掌":
                            # 水平翻转吉祥物
                            print("[成就系统] 应用'鼓掌'风格到Opai")
                            mascot = mascot.transpose(Image.FLIP_LEFT_RIGHT)
                        elif mascot_style == "学术帽":
                            try:
                                print("[成就系统] 在Opai头上添加学术帽")
                                # 在吉祥物头上添加学士帽
                                draw_mascot = ImageDraw.Draw(mascot)
                                # 画一个简易的学士帽
                                cap_x, cap_y = mascot_width//2, mascot_height//5
                                cap_size = mascot_width//3
                                draw_mascot.polygon(
                                    [(cap_x-cap_size, cap_y), (cap_x+cap_size, cap_y), (cap_x, cap_y-cap_size)],
                                    fill=(0, 0, 0, 200)
                                )
                                # 帽子底座
                                draw_mascot.rectangle(
                                    [(cap_x-cap_size, cap_y), (cap_x+cap_size, cap_y+cap_size//2)],
                                    fill=(0, 0, 0, 200)
                                )
                                # 帽穗
                                draw_mascot.line(
                                    [(cap_x, cap_y-cap_size), (cap_x+cap_size, cap_y+cap_size//2)],
                                    fill=(255, 215, 0, 200),
                                    width=2
                                )
                            except Exception as e:
                                print(f"[成就系统] 给Opai添加学术帽失败: {e}")
                        elif mascot_style == "随机":
                            # 随机选择一种风格
                            random_style = np.random.choice(["欢呼", "鼓掌", "学术帽", "正常"])
                            print(f"[成就系统] 随机选择'{random_style}'风格应用到Opai")
                            if random_style == "欢呼":
                                mascot = mascot.rotate(15, resample=Image.BICUBIC, expand=True)
                            elif random_style == "鼓掌":
                                mascot = mascot.transpose(Image.FLIP_LEFT_RIGHT)
                            elif random_style == "学术帽":
                                try:
                                    draw_mascot = ImageDraw.Draw(mascot)
                                    cap_x, cap_y = mascot_width//2, mascot_height//5
                                    cap_size = mascot_width//3
                                    draw_mascot.polygon(
                                        [(cap_x-cap_size, cap_y), (cap_x+cap_size, cap_y), (cap_x, cap_y-cap_size)],
                                        fill=(0, 0, 0, 200)
                                    )
                                    draw_mascot.rectangle(
                                        [(cap_x-cap_size, cap_y), (cap_x+cap_size, cap_y+cap_size//2)],
                                        fill=(0, 0, 0, 200)
                                    )
                                    draw_mascot.line(
                                        [(cap_x, cap_y-cap_size), (cap_x+cap_size, cap_y+cap_size//2)],
                                        fill=(255, 215, 0, 200),
                                        width=2
                                    )
                                except Exception as e:
                                    print(f"[成就系统] 给Opai添加学术帽失败: {e}")
                        
                        # 放置吉祥物
                        certificate.paste(mascot, (width - mascot_width - 50, height - mascot_height - 50), mascot)
                        print("[成就系统] Opai吉祥物成功添加到证书")
                        
                    except Exception as e:
                        print(f"[成就系统] 无法加载Opai吉祥物图片: {e}")
                else:
                    print(f"[成就系统] Opai吉祥物图片不存在: {mascot_path}")
        
        # 如果提供了自定义图像，将其添加到证书中
        if custom_image is not None:
            try:
                # 将输入图像转换为PIL图像
                # 检查输入类型，支持PyTorch张量或NumPy数组
                if TORCH_AVAILABLE and isinstance(custom_image, torch.Tensor):
                    # 如果是PyTorch张量，转换为NumPy
                    # 取第一个批次的图像
                    np_img = custom_image[0].cpu().numpy()
                    # 确保值在0-1范围内
                    if np_img.max() <= 1.0:
                        np_img = (np_img * 255).astype(np.uint8)
                    else:
                        np_img = np_img.astype(np.uint8)
                    pil_image = Image.fromarray(np_img)
                else:
                    # 如果是NumPy数组
                    np_img = custom_image[0]
                    # 确保值在0-1范围内
                    if np_img.max() <= 1.0:
                        np_img = (np_img * 255).astype(np.uint8)
                    else:
                        np_img = np_img.astype(np.uint8)
                    pil_image = Image.fromarray(np_img)
                
                # 根据图像纵横比调整大小
                custom_width = 200
                custom_height = int(pil_image.height * custom_width / pil_image.width)
                
                pil_image = pil_image.resize((custom_width, custom_height), Image.LANCZOS)
                
                # 放在左下角
                certificate.paste(pil_image, (50, height - custom_height - 50))
            except Exception as e:
                print(f"[成就系统] 无法处理自定义图像: {e}")
                
        # 如果是卡通风格，添加一些装饰元素
        if certificate_style == "卡通":
            try:
                # 添加卡通星星
                draw = ImageDraw.Draw(certificate)
                for i in range(10):
                    x = np.random.randint(50, width-50)
                    y = np.random.randint(50, height-50)
                    size = np.random.randint(10, 30)
                    # 绘制简单的星星
                    star_points = []
                    for j in range(5):
                        # 外点
                        angle = j * 2 * np.pi / 5
                        px = x + size * np.cos(angle)
                        py = y + size * np.sin(angle)
                        star_points.append((px, py))
                        # 内点
                        angle += np.pi / 5
                        px = x + size/2 * np.cos(angle)
                        py = y + size/2 * np.sin(angle)
                        star_points.append((px, py))
                    
                    # 随机星星颜色
                    star_color = (
                        np.random.randint(200, 255),
                        np.random.randint(200, 255),
                        np.random.randint(0, 100),
                        200
                    )
                    draw.polygon(star_points, fill=star_color)
            except Exception as e:
                print(f"[成就系统] 添加卡通装饰元素时出错: {e}")
        
        # 根据不同等级添加不同的装饰元素
        try:
            draw = ImageDraw.Draw(certificate)
            
            if level in ["大师", "宗师"]:
                # 高级别添加金色装饰角
                corner_size = 60
                # 左上角
                for i in range(corner_size):
                    draw.line([(0, i), (i, i)], fill=(212, 175, 55, 150), width=2)
                    draw.line([(i, 0), (i, i)], fill=(212, 175, 55, 150), width=2)
                # 右上角
                for i in range(corner_size):
                    draw.line([(width-i, i), (width, i)], fill=(212, 175, 55, 150), width=2)
                    draw.line([(width-i, 0), (width-i, i)], fill=(212, 175, 55, 150), width=2)
                # 左下角
                for i in range(corner_size):
                    draw.line([(0, height-i), (i, height-i)], fill=(212, 175, 55, 150), width=2)
                    draw.line([(i, height-i), (i, height)], fill=(212, 175, 55, 150), width=2)
                # 右下角
                for i in range(corner_size):
                    draw.line([(width-i, height-i), (width, height-i)], fill=(212, 175, 55, 150), width=2)
                    draw.line([(width-i, height-i), (width-i, height)], fill=(212, 175, 55, 150), width=2)
        except Exception as e:
            print(f"[成就系统] 添加等级装饰元素时出错: {e}")
        
        return certificate

class AchievementInfo:
    """显示成就信息的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_achievement_info"
    CATEGORY = "学习中心/成就"
    
    def get_achievement_info(self):
        # 获取用户进度
        progress_file = os.path.join(current_dir, "user_progress", "progress.json")
        completed_count = 0
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8-sig") as f:
                    progress_data = json.load(f)
                    completed_chapters = progress_data.get("completed_chapters", {})
                    completed_count = len(completed_chapters)
            except Exception as e:
                print(f"[成就系统] 无法读取用户进度: {e}")
        
        # 确定成就级别
        level = "初学者"  # 默认级别
        for level_name, required_count in sorted(ACHIEVEMENT_LEVELS.items(), key=lambda x: x[1], reverse=True):
            if completed_count >= required_count:
                level = level_name
                break
        
        # 获取下一个成就级别和需要完成的章节数
        next_level = None
        chapters_needed = 0
        
        # 获取所有级别，按照所需章节数升序排序
        sorted_levels = sorted(ACHIEVEMENT_LEVELS.items(), key=lambda x: x[1])
        
        for level_name, required_count in sorted_levels:
            if required_count > completed_count:
                next_level = level_name
                chapters_needed = required_count - completed_count
                break
        
        # 格式化输出信息
        info = f"当前成就级别: {level}\n"
        info += f"已完成章节数: {completed_count}\n"
        
        if next_level:
            info += f"下一级别: {next_level}（还需完成 {chapters_needed} 个章节）\n"
        else:
            info += "恭喜！你已经达到最高级别！\n"
        
        # 显示所有级别所需的章节数
        info += "\n成就级别一览:\n"
        for level_name, required_count in sorted_levels:
            info += f"- {level_name}: 需要完成 {required_count} 个章节\n"
        
        return (info,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AchievementCertificate": AchievementCertificate,
    "AchievementInfo": AchievementInfo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AchievementCertificate": "成就证书生成器",
    "AchievementInfo": "成就信息显示"
} 