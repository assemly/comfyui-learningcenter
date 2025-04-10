# ComfyUI学习中心成就系统

## 概述

学习中心成就系统提供两个自定义节点：
1. **成就信息显示**：显示用户当前的成就级别、已完成章节数和下一级别信息
2. **成就证书生成器**：生成可自定义的成就证书图像

这些节点可以帮助用户追踪学习进度，并获得完成教程章节的成就感。

## 使用方法

### 成就信息显示

1. 在ComfyUI中添加"成就信息显示"节点
2. 将输出连接到"Text"或类似文本显示节点来查看结果
3. 无需任何输入参数

输出内容包括：
- 当前成就级别
- 已完成章节数
- 下一级别及需完成的章节数
- 所有成就级别的完成要求

### 成就证书生成器

1. 在ComfyUI中添加"成就证书生成器"节点
2. 设置以下参数：
   - **custom_text**: 自定义文本，会显示在证书中部
   - **font_size**: 文本字体大小
   - **text_color**: 文本颜色
   - **bg_color**: 背景颜色
   - **show_mascot**: 是否显示吉祥物角色
   - **border_style**: 证书边框样式
3. 可选：连接其他图像节点到custom_image输入
4. 将输出连接到"PreviewImage"节点查看结果
5. 也可以将输出连接到保存图像的节点

## 成就级别

系统根据用户完成的章节数量自动确定以下成就级别：

- **初学者**: 完成至少1个章节
- **学徒**: 完成至少5个章节
- **实践者**: 完成至少10个章节
- **专家**: 完成至少20个章节
- **大师**: 完成至少30个章节
- **宗师**: 完成至少50个章节

## 常见问题与解决方法

### 1. 与PreviewImage节点兼容性问题

**症状**: 使用PreviewImage节点时出现以下错误：
```
AttributeError: 'numpy.ndarray' object has no attribute 'cpu'
```

**解决方法**:
- 安装PyTorch库可以解决此问题：`pip install torch`
- 或者使用SaveImage节点代替PreviewImage节点

### 2. 字体显示问题

**症状**: 证书上的文字显示为默认字体或乱码

**解决方法**:
- 确保系统中安装了中文字体（如微软雅黑或黑体）
- 尝试减小字体大小

### 3. 吉祥物图像未显示

**症状**: 开启show_mascot选项但未显示吉祥物

**解决方法**:
- 确认resources目录中存在opai.png文件
- 检查日志中是否有关于图像加载的错误信息

## 技术细节

成就系统将用户完成的章节记录存储在user_progress/progress.json文件中。当用户标记章节为已完成时，系统会自动更新此文件。成就系统会读取此文件来确定用户的成就级别。

证书生成器使用PIL库创建图像，并根据ComfyUI环境自动选择返回PyTorch张量或NumPy数组格式。 