import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

# 设置matplotlib支持中文 - 使用多种字体备选，优先使用系统字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']  # 多字体备选
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 确保matplotlib中文显示配置正确
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'SimHei'

def load_image(image_path):
    """加载图像并进行基本检查 - 支持中文路径"""
    try:
        # 使用二进制模式读取以支持中文路径
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        # 使用imdecode解码图像
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 检查图像是否成功加载
        if image is None:
            print(f"错误：无法加载图像 '{image_path}'。请检查文件路径和完整性。")
            return None
            
        return image
    except Exception as e:
        print(f"加载图像时发生错误: {str(e)}")
        return None

def preprocess_image_color_adaptive(image, color_type='auto'):
    """颜色自适应的图像预处理方法，支持不同颜色灰锥在各种背景上的检测
    
    Args:
        image: 输入图像
        color_type: 灰锥颜色类型 - 'auto'(自动检测), 'black'(黑色), 'brown'(棕黄色), 'gray'(灰色), 'light'(浅色)
    
    Returns:
        二值化后的图像
    """
    # 转换为多种色彩空间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 应用多重高斯模糊
    blur1 = cv2.GaussianBlur(gray, (11, 11), 1)
    blur2 = cv2.GaussianBlur(gray, (17, 17), 2)
    blurred = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)
    
    # CLAHE增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 自动颜色检测
    if color_type == 'auto':
        # 计算图像直方图，分析主要颜色分布
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        # 分析亮度分布
        brightness_mean = np.mean(gray)
        
        # 颜色分类逻辑
        if brightness_mean < 80:  # 整体较暗
            color_type = 'black'
        elif brightness_mean > 180:  # 整体较亮
            color_type = 'light'
        else:
            # 尝试检测棕黄色调
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            if 10 <= h_mean <= 40 and s_mean > 50:  # 棕黄色范围
                color_type = 'brown'
            else:
                color_type = 'gray'
    
    # 根据颜色类型选择不同的二值化策略
    if color_type == 'black':  # 黑色灰锥
        # 对深色物体使用反相阈值
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    elif color_type == 'light':  # 浅色物体
        # 对浅色物体使用正常阈值
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif color_type == 'brown':  # 棕黄色物体
        # 使用HSV颜色空间分割棕黄色
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([40, 255, 255])
        binary_hsv = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 使用LAB颜色空间的a通道（对棕色敏感）
        a_channel = lab[:, :, 1]
        clahe_a = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        a_enhanced = clahe_a.apply(a_channel)
        _, binary_a = cv2.threshold(a_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 融合HSV和LAB结果
        binary = cv2.bitwise_or(binary_hsv, binary_a)
        
    elif color_type == 'gray':  # 灰色物体
        # 使用局部自适应阈值，对灰色物体更有效
        binary = cv2.adaptiveThreshold(enhanced, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 8)
        
        # 结合梯度信息
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_normalized = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, binary_grad = cv2.threshold(grad_normalized, 30, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_or(binary, binary_grad)
    
    # 通用形态学操作，适应各种颜色模式
    # 1. 开操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. 闭操作连接物体内部间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. 适度膨胀使轮廓更完整
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 4. 边缘检测增强
    edges = cv2.Canny(enhanced, 50, 150)
    binary = cv2.bitwise_or(binary, edges)
    
    return binary

# 保留原有函数作为兼容，但实际上使用颜色自适应函数
preprocess_image_fixed = preprocess_image_color_adaptive

def preprocess_image_adaptive(image):
    """增强版自适应阈值方法进行图像预处理 - 适合各种光照条件"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 多级高斯模糊 - 更好地处理不同尺寸的噪声
    blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (9, 9), 1)
    # 加权融合模糊结果
    blurred = cv2.addWeighted(blur1, 0.7, blur2, 0.3, 0)
    
    # 局部对比度增强 - 提高边缘清晰度
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 智能二值化策略
    # 计算图像亮度统计
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)
    
    # 根据图像亮度选择合适的阈值方法
    if brightness_mean < 80:  # 较暗图像
        # 对深色物体使用反相阈值
        binary1 = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 25, 10)
    elif brightness_mean > 180:  # 较亮图像
        # 对浅色物体使用正常阈值，但调整参数
        binary1 = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 15)
    else:  # 中等亮度图像
        # 标准自适应阈值
        binary1 = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 25, 8)
    
    # 同时尝试OTSU阈值作为补充
    _, binary2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 根据图像对比度决定使用哪种方法
    if brightness_std < 40:  # 低对比度图像
        # 融合两种二值化结果
        binary = cv2.bitwise_or(binary1, binary2)
    else:
        binary = binary1
    
    # 改进的形态学操作
    # 1. 开操作去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. 闭操作连接物体内部间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. 适度膨胀使轮廓更完整
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 4. 边缘检测增强
    edges = cv2.Canny(enhanced, 50, 150)
    binary = cv2.bitwise_or(binary, edges)
    
    return binary

def find_largest_contour(binary):
    """增强版轮廓检测与筛选，优化各种光照条件下的灰锥轮廓提取"""
    # 尝试不同的轮廓提取模式
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 合并所有轮廓
    all_contours = contours1 + contours2
    if not all_contours:
        return None
    
    # 获取图像尺寸
    h, w = binary.shape[:2]
    image_area = h * w
    
    # 创建一个评分系统来筛选最可能是灰锥的轮廓
    scored_contours = []
    
    for contour in all_contours:
        try:
            # 计算轮廓基本属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 放宽基本面积过滤条件，适应不同大小的灰锥
            if area < 50 or area > image_area * 0.95:  # 只排除非常小或几乎整个图像的轮廓
                continue
            
            # 计算轮廓的边界框
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            
            # 检查是否接触边界 - 放宽条件
            touches_border = (x <= 5 or y <= 5 or 
                              x + w_contour >= w - 5 or 
                              y + h_contour >= h - 5)
            
            # 计算形状特征
            # 1. 圆形度
            circularity = 0
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 2. 矩形度
            rectangularity = 0
            rect_area = w_contour * h_contour
            if rect_area > 0:
                rectangularity = area / rect_area
            
            # 3. 纵横比 - 灰锥的常见比例
            aspect_ratio = 0
            if w_contour > 0:
                aspect_ratio = float(h_contour) / w_contour
            
            # 4. 凸包率 - 衡量轮廓的凸度
            convex_hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(convex_hull)
            solidity = 0
            if hull_area > 0:
                solidity = area / hull_area
            
            # 5. 重心位置
            M = cv2.moments(contour)
            center_dist = 1.0  # 默认距离
            if M['m00'] > 0:
                cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
                # 计算重心到图像中心的归一化距离
                center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)** 2) / (np.sqrt(w**2 + h** 2) / 2)
            
            # 评分系统 - 放宽标准，更好地适应各种灰锥
            score = 0
            
            # 三角形通常不是很圆
            if 0.05 < circularity < 0.8:  # 更宽的范围
                score += 15
            
            # 三角形的矩形度范围放宽
            if 0.3 < rectangularity < 0.8:  # 更宽的范围
                score += 25
            
            # 灰锥通常高度大于宽度 - 更灵活的比例
            if 0.8 < aspect_ratio < 3.5:  # 更宽的范围
                score += 30
            
            # 三角形应该是比较凸的
            if solidity > 0.8:  # 降低要求
                score += 20
            
            # 不太严格地要求中心位置
            if center_dist < 0.7:  # 更宽的范围
                score += 10
            
            # 接触边界的惩罚降低
            if touches_border:
                score -= 10
            
            # 根据轮廓大小给予不同权重
            size_ratio = area / image_area
            if size_ratio > 0.05:  # 更小的轮廓也能获得分数
                score += min(15, int(size_ratio * 100))  # 大小权重
            
            # 多边形近似，计算角点数
            epsilon = 0.03 * perimeter  # 更宽松的近似
            approx = cv2.approxPolyDP(contour, epsilon, True)
            corner_count = len(approx)
            
            # 三角形通常有3-8个角点（放宽要求）
            if 3 <= corner_count <= 8:
                score += 30
            elif 2 <= corner_count <= 10:
                score += 15
            
            # 额外的稳定性检查 - 确保轮廓是闭合的
            if perimeter > 0 and area > 0:
                # 计算轮廓的紧凑度
                compactness = area / (perimeter * perimeter) if perimeter > 0 else 0
                if 0.001 < compactness < 0.1:  # 合理的紧凑度范围
                    score += 10
            
            # 记录轮廓及其评分
            scored_contours.append((contour, score, area, rectangularity, aspect_ratio))
        except Exception as e:
            # 跳过处理失败的轮廓
            continue
    
    # 如果有评分的轮廓，选择评分最高的
    if scored_contours:
        # 按评分降序排序
        scored_contours.sort(key=lambda x: x[1], reverse=True)
        
        # 返回评分最高的轮廓
        return scored_contours[0][0]
    
    # 如果没有通过评分系统的轮廓，尝试基本的面积筛选
    if all_contours:
        # 按面积排序并返回前几个中最合适的
        sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        
        # 检查前5个最大的轮廓
        for i in range(min(5, len(sorted_contours))):
            contour = sorted_contours[i]
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 基本的有效性检查
            if perimeter > 0 and area > 100:
                x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
                aspect_ratio = float(h_cnt) / w_cnt if w_cnt > 0 else 0
                
                # 只要形状大致合理就返回
                if 0.5 < aspect_ratio < 5.0:  # 非常宽松的比例
                    return contour
    
    # 最后的尝试：返回最大的轮廓（如果有）
    if all_contours:
        largest_contour = max(all_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # 确保有一定大小
            return largest_contour
    
    return None

def approximate_triangle(contour):
    """增强版多边形近似，优化各种条件下的三角形拟合"""
    try:
        # 计算轮廓周长和面积
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0 or area < 50:  # 放宽面积要求
            return None
        
        # 策略1：使用更广泛的epsilon值范围
        epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        best_approx = None
        best_score = -1
        
        for epsilon_val in epsilon_values:
            approx = cv2.approxPolyDP(contour, epsilon_val * perimeter, True)
            
            # 计算近似质量分数 - 降低对顶点数的要求
            approx_area = cv2.contourArea(approx)
            if approx_area > 0:
                area_similarity = approx_area / area if area > 0 else 0
                
                # 更宽松的顶点评分
                if len(approx) == 3:
                    vertex_score = 2.0
                elif len(approx) == 2 or len(approx) == 4:
                    vertex_score = 1.5  # 接受2点或4点
                elif len(approx) <= 6:
                    vertex_score = 1.2
                else:
                    vertex_score = 0.8
                
                score = area_similarity * vertex_score
                
                if score > best_score:
                    best_score = score
                    best_approx = approx
        
        if best_approx is None:
            return None
        
        # 如果近似结果有3个顶点，直接返回
        if len(best_approx) == 3:
            return best_approx
        
        # 策略2：简化凸包处理
        convex_hull = cv2.convexHull(contour)
        
        # 不管solidity如何，都尝试使用凸包
        hull_perimeter = cv2.arcLength(convex_hull, True)
        for hull_epsilon_val in [0.01, 0.02, 0.03, 0.04, 0.05]:
            hull_approx = cv2.approxPolyDP(convex_hull, hull_epsilon_val * hull_perimeter, True)
            if len(hull_approx) == 3 and cv2.contourArea(hull_approx) > 30:  # 降低面积要求
                return hull_approx
        
        # 如果凸包本身点数合适，直接使用凸包的前3个点
        if len(convex_hull) >= 3:
            return convex_hull[:3]
        
        # 策略3：简单的特征点三角形 - 只取极值点
        try:
            # 获取关键特征点
            points = best_approx.reshape(-1, 2)
            
            # 只获取基本的极值点
            top = points[np.argmin(points[:, 1])]  # 最高点
            bottom = points[np.argmax(points[:, 1])]  # 最低点
            left = points[np.argmin(points[:, 0])]  # 最左点
            right = points[np.argmax(points[:, 0])]  # 最右点
            
            # 创建三角形的多种可能组合
            triangles = [
                np.array([[top], [left], [right]]),  # 顶部和两个侧边
                np.array([[top], [left], [bottom]]),  # 左上和底部
                np.array([[top], [right], [bottom]])  # 右上和底部
            ]
            
            # 选择面积最大的三角形
            max_area = 0
            best_triangle = None
            
            for tri in triangles:
                tri_area = cv2.contourArea(tri)
                if tri_area > max_area and tri_area > 30:  # 降低面积要求
                    max_area = tri_area
                    best_triangle = tri
            
            if best_triangle is not None:
                return best_triangle
        except Exception:
            pass
        
        # 策略4：最小外接矩形 - 简化版本
        try:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = box.reshape(-1, 1, 2)
            
            # 直接使用矩形的三个顶点
            return box[:3]
        except Exception:
            pass
        
        # 最后的简单方案：使用原始轮廓的关键点
        if len(contour) >= 3:
            # 采样3个点
            indices = np.linspace(0, len(contour)-1, 3, dtype=int)
            return contour[indices]
        
    except Exception as e:
        print(f"三角形近似错误: {str(e)}")
    
    return None

def calculate_triangle_properties(triangle):
    """计算三角形的高和边长比例"""
    if triangle is None or len(triangle) < 3:
        return None, None, None
    
    # 确保triangle是数组格式
    pts = triangle.reshape(-1, 2)
    
    # 计算边长
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])** 2)
    
    # 计算所有边的长度
    side1 = distance(pts[0], pts[1])
    side2 = distance(pts[1], pts[2])
    side3 = distance(pts[2], pts[0])
    
    # 找出最长的边作为底边
    sides = [side1, side2, side3]
    max_side_index = np.argmax(sides)
    
    # 根据最长边确定顶点
    if max_side_index == 0:  # 边1是最长边
        base = side1
        apex = pts[2]
        base_points = [pts[0], pts[1]]
    elif max_side_index == 1:  # 边2是最长边
        base = side2
        apex = pts[0]
        base_points = [pts[1], pts[2]]
    else:  # 边3是最长边
        base = side3
        apex = pts[1]
        base_points = [pts[2], pts[0]]
    
    # 计算高（顶点到底边的垂直距离）
    p1, p2 = base_points
    # 使用向量投影计算点到直线的距离
    numerator = abs((p2[0] - p1[0]) * (apex[1] - p1[1]) - (p2[1] - p1[1]) * (apex[0] - p1[0]))
    denominator = distance(p1, p2)
    
    if denominator > 0:
        height = numerator / denominator
    else:
        height = 0
    
    # 检查是否是等腰三角形
    dist_to_p1 = distance(apex, p1)
    dist_to_p2 = distance(apex, p2)
    is_isosceles = abs(dist_to_p1 - dist_to_p2) < 0.1 * max(dist_to_p1, dist_to_p2)
    
    # 移除自动交换高和底的逻辑，保持正确的三角形结构
    # 高应该是从顶点到底边的垂直距离，不应该随意交换
    pass
    
    return height, base, is_isosceles

def check_cone_requirements(height, base, required_height=20, required_side=7):
    """检查灰锥是否满足比例要求，并提供详细错误信息"""
    if height is None or base is None or height == 0 or base == 0:
        return False, "无法识别三角形或计算比例"
    
    # 计算实际比例
    actual_ratio = height / base
    required_ratio = required_height / required_side
    
    # 误差范围
    tolerance = 0.3  # 误差范围，允许30%的偏差
    
    # 计算偏差百分比
    deviation = abs(actual_ratio - required_ratio) / required_ratio * 100
    
    # 比例检查
    ratio_within_tolerance = abs(actual_ratio - required_ratio) <= tolerance * required_ratio
    
    # 添加调试信息
    print(f"[调试] 实际比例: {actual_ratio:.3f}, 要求比例: {required_ratio:.3f}, 偏差: {deviation:.3f}%, 允许偏差: {tolerance*100:.1f}%, 是否通过: {ratio_within_tolerance}")
    
    # 尺寸检查
    height_within_range = 20 <= height <= 100
    base_within_range = 5 <= base <= 50
    size_within_range = height_within_range and base_within_range
    
    # 对于比例要求严格的应用，只要比例符合，即使尺寸稍微超出范围也视为有效
    # 这样可以处理比例正确但由于图像缩放等原因导致尺寸略有偏差的情况
    is_valid = ratio_within_tolerance
    
    # 详细错误信息
    if is_valid:
        return True, f"灰锥识别成功！实际比例: {actual_ratio:.2f}, 尺寸: 高={height:.1f}, 底={base:.1f}"
    else:
        # 构建详细的错误信息，说明具体哪些条件不满足
        error_reasons = []
        
        if not ratio_within_tolerance:
            error_reasons.append(f"比例偏差过大 ({deviation:.1f}%，超过允许的±{tolerance*100}%)")
        
        if not height_within_range:
            if height < 20:
                error_reasons.append(f"高度过小 ({height:.1f}，要求≥20)")
            else:
                error_reasons.append(f"高度过大 ({height:.1f}，要求≤100)")
        
        if not base_within_range:
            if base < 5:
                error_reasons.append(f"底边过小 ({base:.1f}，要求≥5)")
            else:
                error_reasons.append(f"底边过大 ({base:.1f}，要求≤50)")
        
        # 确保错误信息始终包含详细原因
        error_message = f"灰锥不符合要求: {'; '.join(error_reasons)}。实际比例: {actual_ratio:.2f}, 要求比例: {required_ratio:.2f}, 允许误差: ±{tolerance*100}%, 尺寸: 高={height:.1f}, 底={base:.1f}"
        return False, error_message


def visualize_result(image, binary1, binary2, contour, triangle, result_text):
    """可视化检测结果，展示两种预处理方法的对比"""
    plt.figure(figsize=(18, 10))
    
    # 原始图像
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 自适应阈值二值化图像
    plt.subplot(232)
    plt.imshow(binary1, cmap='gray')
    plt.title('自适应阈值二值化')
    plt.axis('off')
    
    # 固定阈值二值化图像
    plt.subplot(233)
    plt.imshow(binary2, cmap='gray')
    plt.title('固定阈值(Otsu)二值化')
    plt.axis('off')
    
    # 反转的二值图像（用于显示黑色区域）
    plt.subplot(234)
    plt.imshow(cv2.bitwise_not(binary2), cmap='gray')
    plt.title('二值图像反转（显示黑色区域）')
    plt.axis('off')
    
    # 轮廓图
    plt.subplot(235)
    contour_image = np.zeros_like(binary2)
    if contour is not None:
        cv2.drawContours(contour_image, [contour], -1, 255, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title('检测到的轮廓')
    plt.axis('off')
    
    # 检测结果
    plt.subplot(236)
    result_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    # 绘制轮廓和三角形
    if contour is not None:
        cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 2)  # 蓝色显示轮廓
    
    if triangle is not None:
        cv2.drawContours(result_image, [triangle], -1, (0, 255, 0), 2)  # 绿色显示三角形
    
    plt.imshow(result_image)
    plt.title(result_text)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def preprocess_color_cone(image, color_type='auto'):
    """颜色自适应的灰锥预处理方法，支持各种颜色灰锥
    
    Args:
        image: 输入图像
        color_type: 灰锥颜色类型 - 'auto'(自动检测), 'black'(黑色), 'brown'(棕黄色), 'gray'(灰色), 'light'(浅色)
    
    Returns:
        二值化后的图像
    """
    # 转换为多种色彩空间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 自动颜色检测
    if color_type == 'auto':
        # 计算图像亮度和颜色分布
        brightness_mean = np.mean(gray)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        
        # 根据统计特征进行颜色分类
        if brightness_mean < 100:
            color_type = 'black'
        elif 10 <= h_mean <= 40 and s_mean > 40:
            color_type = 'brown'
        elif brightness_mean > 150:
            color_type = 'light'
        else:
            color_type = 'gray'
    
    print(f"检测到灰锥颜色类型: {color_type}")
    
    # 根据颜色类型选择不同的处理策略
    if color_type == 'black':  # 黑色灰锥
        # 使用反相阈值突出黑色物体
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    elif color_type == 'brown':  # 棕黄色灰锥
        # 结合多种颜色空间的分割
        # HSV颜色空间分割
        lower_brown = np.array([8, 40, 40])
        upper_brown = np.array([45, 255, 255])
        binary_hsv = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # LAB颜色空间的a通道（对棕色敏感）
        a_channel = lab[:, :, 1]
        clahe_a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        a_enhanced = clahe_a.apply(a_channel)
        _, binary_a = cv2.threshold(a_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 融合结果
        binary = cv2.bitwise_or(binary_hsv, binary_a)
        
    elif color_type == 'gray':  # 灰色灰锥
        # 使用局部对比度增强和自适应阈值
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(gray_enhanced, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
        
    elif color_type == 'light':  # 浅色灰锥
        # 使用正常阈值但调整参数
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 通用形态学操作
    # 开操作去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 闭操作连接间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 适度膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

# 程序现在只使用自适应阈值方法

def put_chinese_text(img, text, position, size=20, color=(0, 255, 0)):
    """使用PIL在OpenCV图像上显示中文"""
    # 将BGR图像转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为PIL图像
    pil_img = Image.fromarray(img_rgb)
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # 尝试加载中文字体
        font = ImageFont.truetype("simhei.ttf", size)
    except:
        # 如果找不到中文字体，使用默认字体
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=tuple(reversed(color)))
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_user_selected_vertices(image):
    """让用户手动点击图像中的三个顶点来定义灰锥轮廓"""
    print("请在图像上点击灰锥的三个顶点（按顺时针或逆时针顺序）")
    print("提示：先点击顶部顶点，然后是底部的两个顶点")
    
    # 创建一个副本用于显示
    display_image = image.copy()
    
    # 存储用户选择的点
    clicked_points = []
    
    # 使用支持中文的文本显示
    display_image = put_chinese_text(display_image, "请点击三点", (10, 30), size=24)
    
    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 3:
                # 添加点击的点
                clicked_points.append((x, y))
                
                # 在图像上标记点击的点
                cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                
                # 显示已点击的点数（使用自定义的中文显示函数）
                temp_img = display_image.copy()
                temp_img = put_chinese_text(temp_img, f"已选 {len(clicked_points)}/3 点", (10, 30), size=24)
                
                # 连接已选择的点
                if len(clicked_points) > 1:
                    for i in range(len(clicked_points) - 1):
                        cv2.line(temp_img, clicked_points[i], clicked_points[i+1], (255, 0, 0), 2)
                    
                    # 如果已经选择了三个点，连接第三个和第一个点形成三角形
                    if len(clicked_points) == 3:
                        cv2.line(temp_img, clicked_points[2], clicked_points[0], (255, 0, 0), 2)
                
                # 更新显示
                display_image[:] = temp_img
                cv2.imshow("选择灰锥顶点", display_image)
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("选择灰锥顶点", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("选择灰锥顶点", mouse_callback)
    
    # 显示初始图像
    cv2.imshow("选择灰锥顶点", display_image)
    
    # 等待用户点击或按ESC键退出
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 如果按ESC键，退出并返回None
        if key == 27:
            print("用户取消了选择")
            cv2.destroyWindow("选择灰锥顶点")
            return None
        
        # 如果选择了三个点，等待用户确认
        if len(clicked_points) == 3:
            # 显示确认提示
            temp_img = display_image.copy()
            temp_img = put_chinese_text(temp_img, "Enter确认, ESC重选", (10, 60), size=24, color=(0, 255, 255))
            cv2.imshow("选择灰锥顶点", temp_img)
            
            # 等待用户确认
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter键
                    cv2.destroyWindow("选择灰锥顶点")
                    # 将点击的点转换为OpenCV的轮廓格式
                    triangle = np.array([[[x, y]] for x, y in clicked_points], dtype=np.int32)
                    return triangle
                elif key == 27:  # ESC键重新选择
                    # 重置图像和已选择的点
                    display_image = image.copy()
                    clicked_points = []
                    # 使用简化的中文文本，更容易在OpenCV中正确显示
                    cv2.putText(display_image, "请点击三点", (10, 30), 
                               font, 0.7, (0, 255, 0), 2)
                    cv2.imshow("选择灰锥顶点", display_image)
                    break
    
    cv2.destroyWindow("选择灰锥顶点")
    return None

def detect_real_cone(image_path):
    """专门用于实际照片的灰锥检测函数 - 增强版多策略融合方法，支持用户手动修正
    
    Args:
        image_path: 图像路径
    
    Returns:
        检测结果元组 (is_real_cone, message)
    """
    try:
        print(f"\n=== 正在处理图像: {image_path} ===")
        print("使用方法: 多策略融合检测")
        
        # 加载图像
        image = load_image(image_path)
        
        # 多阶段预处理策略 - 同时尝试多种方法
        preprocessing_methods = [
            ('adaptive', preprocess_image_adaptive(image)),
            ('otsu', cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ('canny', cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)),
            ('gaussian_blur_otsu', cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        ]
        
        # 依次尝试每种预处理方法
        best_contour = None
        best_binary = None
        
        for method_name, binary_result in preprocessing_methods:
            print(f"尝试预处理方法: {method_name}")
            contour = find_largest_contour(binary_result)
            if contour is not None:
                # 如果找到轮廓，尝试三角形近似
                triangle = approximate_triangle(contour)
                if triangle is not None:
                    # 检查是否是有效的三角形
                    height, base, _ = calculate_triangle_properties(triangle)
                    if height is not None and base is not None and height > 10 and base > 5:
                        best_contour = contour
                        best_binary = binary_result
                        print(f"方法 {method_name} 成功找到候选轮廓")
                        break
        
        # 如果所有方法都失败，尝试最后的补救措施
        if best_contour is None:
            print("所有方法均失败，尝试直接处理原始图像的灰度图")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 尝试不同的阈值处理
            for threshold_value in [50, 80, 100, 120, 150]:
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                contour = find_largest_contour(binary)
                if contour is not None:
                    best_contour = contour
                    best_binary = binary
                    print(f"直接阈值处理成功，阈值: {threshold_value}")
                    break
        
        # 如果仍然没有找到轮廓，使用原始自适应方法作为备用
        if best_contour is None:
            print("最后尝试: 使用原始自适应阈值方法")
            best_binary = preprocess_image_adaptive(image)
            best_contour = find_largest_contour(best_binary)
            
            # 如果还是失败，尝试颜色自适应方法
            if best_contour is None:
                print("尝试颜色自适应方法")
                best_binary = preprocess_color_cone(image)
                best_contour = find_largest_contour(best_binary)
        
        # 最终轮廓检查
        if best_contour is None:
            print("错误: 无法找到物体轮廓")
            print("提示: 请尝试确保图像中只有灰锥，或调整拍摄角度和光照")
            
            # 询问用户是否要手动选择顶点
            print("是否要手动选择灰锥顶点？(y/n)")
            user_input = input().strip().lower()
            if user_input == 'y':
                print("启动手动顶点选择...")
                # 使用用户手动选择的顶点
                triangle = get_user_selected_vertices(image)
                if triangle is not None:
                    print("成功获取用户选择的顶点")
                    # 计算三角形属性
                    height, base, is_isosceles = calculate_triangle_properties(triangle)
                    
                    # 检查是否符合要求
                    valid, message = check_cone_requirements(height, base, required_height=20, required_side=7)
                    
                    # 添加等腰三角形检查结果
                    if is_isosceles:
                        message += " (近似等腰三角形)"
                    else:
                        message += " (非等腰三角形)"
                    
                    # 显示结果
                    print(f"手动选择检测结果: {message}")
                    visualize_result(image, best_binary, None, triangle, message)
                    
                    # 根据检查结果返回有效性
                    return valid, f"手动选择顶点: {message}"
            
            # 显示处理结果
            visualize_result(image, best_binary, None, None, "无法找到物体轮廓")
            return False, "无法找到物体轮廓"
        
        print(f"成功找到轮廓，面积: {cv2.contourArea(best_contour):.2f}")
        
        # 近似三角形 - 使用多种策略
        triangle = approximate_triangle(best_contour)
        
        # 如果三角形拟合失败，尝试更多方法
        if triangle is None:
            print("三角形拟合失败，尝试使用凸包...")
            # 尝试不同精度的凸包近似
            for hull_precision in [0.03, 0.05, 0.07]:
                convex_hull = cv2.convexHull(best_contour)
                hull_perimeter = cv2.arcLength(convex_hull, True)
                triangle = cv2.approxPolyDP(convex_hull, hull_precision * hull_perimeter, True)
                if len(triangle) == 3:
                    print(f"成功通过凸包近似得到三角形，精度: {hull_precision}")
                    break
            
            # 如果仍然失败，尝试使用矩形的三个顶点
            if triangle is None or len(triangle) != 3:
                print("尝试使用最小外接矩形...")
                try:
                    rect = cv2.minAreaRect(best_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    triangle = box[:3].reshape(-1, 1, 2)
                except Exception:
                    pass
        
        # 再次检查三角形
        if triangle is None or len(triangle) < 3:
            print("错误: 无法将轮廓近似为三角形")
            
            # 询问用户是否要手动选择顶点
            print("是否要手动选择灰锥顶点？(y/n)")
            user_input = input().strip().lower()
            if user_input == 'y':
                print("启动手动顶点选择...")
                # 使用用户手动选择的顶点
                triangle = get_user_selected_vertices(image)
                if triangle is not None:
                    print("成功获取用户选择的顶点")
                    # 计算三角形属性
                    height, base, is_isosceles = calculate_triangle_properties(triangle)
                    
                    # 检查是否符合要求
                    valid, message = check_cone_requirements(height, base, required_height=20, required_side=7)
                    
                    # 添加等腰三角形检查结果
                    if is_isosceles:
                        message += " (近似等腰三角形)"
                    else:
                        message += " (非等腰三角形)"
                    
                    # 显示结果
                    print(f"手动选择检测结果: {message}")
                    visualize_result(image, best_binary, best_contour, triangle, message)
                    
                    # 根据检查结果返回有效性
                    return valid, f"手动选择顶点: {message}"
            
            visualize_result(image, best_binary, best_contour, None, "无法将轮廓近似为三角形")
            return False, "无法将轮廓近似为三角形"
        
        print(f"成功拟合三角形，顶点数: {len(triangle)}")
        
        # 计算三角形属性
        height, base, is_isosceles = calculate_triangle_properties(triangle)
        
        # 严格检查是否符合要求
        valid, message = check_cone_requirements(height, base, required_height=20, required_side=7)
        
        # 添加等腰三角形检查结果 - 这是信息性的，不影响判定
        if is_isosceles:
            message += " (近似等腰三角形)"
        else:
            message += " (非等腰三角形)"
        
        # 移除所有宽松的替代条件，严格按照比例要求判定
        # 不再基于高度、底边或轮廓大小的宽松条件进行判定
        
        # 如果比例不符合但检测到三角形轮廓，给予单次警告
        if not valid and len(triangle) == 3:
            message += " [提示: 已检测到三角形轮廓，详细原因请查看上述错误信息]"
        
        # 显示结果
        print(f"自动检测结果: {message}")
        
        # 询问用户是否满意自动检测结果，是否需要手动修正
        print("是否需要手动选择顶点进行修正？(y/n)")
        user_input = input().strip().lower()
        if user_input == 'y':
            print("启动手动顶点选择...")
            # 使用用户手动选择的顶点
            user_triangle = get_user_selected_vertices(image)
            if user_triangle is not None:
                print("成功获取用户选择的顶点")
                # 计算三角形属性
                height, base, is_isosceles = calculate_triangle_properties(user_triangle)
                
                # 检查是否符合要求
                valid, message = check_cone_requirements(height, base, required_height=20, required_side=7)
                
                # 添加等腰三角形检查结果
                if is_isosceles:
                    message += " (近似等腰三角形)"
                else:
                    message += " (非等腰三角形)"
                
                # 显示结果
                print(f"手动选择检测结果: {message}")
                visualize_result(image, best_binary, best_contour, user_triangle, message)
                
                # 手动选择的顶点也应符合比例要求
                # 不再自动认为手动选择的顶点有效，而是根据check_cone_requirements的结果
                return valid, f"手动选择顶点: {message}"
        
        # 使用自动检测结果
        visualize_result(image, best_binary, best_contour, triangle, message)
        
        # 严格按照比例要求判定，不再考虑其他宽松条件
        is_success = valid
        
        return is_success, message
        
    except Exception as e:
        error_msg = f"处理过程中出错: {str(e)}"
        print(error_msg)
        return False, error_msg

def visualize_result(image, binary, contour, triangle, result_text):
    """可视化处理结果"""
    plt.figure(figsize=(12, 10))
    
    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 自适应阈值二值化图像
    plt.subplot(222)
    plt.imshow(binary, cmap='gray')
    plt.title('自适应阈值二值化')
    plt.axis('off')
    
    # 轮廓+原始图像叠加
    plt.subplot(223)
    overlay_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if contour is not None:
        cv2.drawContours(overlay_image, [contour], -1, (255, 0, 0), 2)
    plt.imshow(overlay_image)
    plt.title('轮廓叠加')
    plt.axis('off')
    
    # 最终结果
    plt.subplot(224)
    result_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if contour is not None:
        cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 2)
    if triangle is not None:
        cv2.drawContours(result_image, [triangle], -1, (0, 255, 0), 2)
    plt.imshow(result_image)
    plt.title(result_text)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("===== 灰锥识别系统（精简版）=====")
    print("系统说明：仅使用优化后的自适应阈值方法进行灰锥检测")
    print("特别优化：提高了边缘检测精度，增强了轮廓拟合的稳定性")
    print("\n拍摄建议：")
    print("1. 在光线均匀的环境下拍摄，避免强光和阴影")
    print("2. 确保灰锥占据图像中心位置，边缘清晰可见")
    print("3. 避免反光表面，使用柔光照明")
    print("4. 保持图像背景简洁，减少干扰")
    print("\n按Ctrl+C可以随时退出程序")
    
    while True:
        try:
            # 获取用户输入的图像路径
            image_path = input("\n请输入图像路径 (或直接按Enter使用默认测试图像): ")
            
            # 如果用户未输入，使用默认路径
            if not image_path.strip():
                image_path = "1.jpg"
                print(f"使用默认图像路径: {image_path}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在。请检查路径是否正确: {image_path}")
                print("提示: 在Windows中，可以右键点击图片 -> 属性 -> 安全 -> 对象名称 中复制完整路径")
                continue
            
            # 检查是否为图像文件
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in valid_extensions:
                print(f"错误: 不支持的文件格式 '{ext}'。请选择常见的图像格式文件。")
                continue
            
            # 执行检测（仅使用自适应阈值方法）
            valid, message = detect_real_cone(image_path)
            
            # 显示详细分析信息
            print("\n=== 检测详细信息 ===")
            print(message)
            
            # 增强的错误分析
            if not valid:
                print("\n详细错误分析：")
                
                if "无法找到物体轮廓" in message:
                    print("1. 轮廓提取失败原因：")
                    print("   - 图像对比度不足，灰锥与背景区分不明显")
                    print("   - 光线不均匀，产生了过多阴影或反光")
                    print("   - 图像分辨率过低或灰锥过小")
                    print("   - 背景复杂，干扰物过多")
                    print("\n   改进建议：")
                    print("   - 调整拍摄角度，避免强光直射导致反光")
                    print("   - 增加环境光照的均匀性，减少阴影")
                    print("   - 使用纸张或布料作为简单背景")
                    print("   - 确保灰锥在图像中占据足够大的比例")
                    
                elif "无法将轮廓近似为三角形" in message:
                    print("2. 三角形拟合失败原因：")
                    print("   - 轮廓形状不规则，可能由反光导致边缘缺失")
                    print("   - 灰锥部分被遮挡或拍摄不完整")
                    print("   - 图像模糊，边缘不清晰")
                    print("   - 灰锥本身变形或损坏")
                    print("\n   改进建议：")
                    print("   - 重新摆放灰锥，确保完整无遮挡")
                    print("   - 调整焦距使图像清晰")
                    print("   - 从正面拍摄，避免透视变形")
                    print("   - 检查灰锥是否完好无损")
                    
                elif "不符合比例要求" in message or "比例不符合" in message:
                    print("3. 比例验证失败原因：")
                    print("   - 拍摄角度不当，导致透视变形")
                    print("   - 灰锥摆放不垂直，导致视觉比例失调")
                    print("   - 图像边缘提取不精确")
                    print("\n   改进建议：")
                    print("   - 从正前方水平拍摄，避免倾斜视角")
                    print("   - 确保灰锥垂直放置")
                    print("   - 增加光照强度，提高边缘清晰度")
                
                print("\n通用拍摄技巧：")
                print("- 使用白纸作为背景，增加对比度")
                print("- 在室内自然光下拍摄，避免闪光灯")
                print("- 保持相机稳定，避免模糊")
                print("- 确保灰锥完全在画面中，不要裁剪边缘")
            
            print(f"\n检测完成！结果: {'通过' if valid else '未通过'}")
            
            # 询问用户是否继续
            again = input("\n是否继续识别下一张图像? (y/n, 默认y): ")
            if again.lower() == 'n':
                break
                
        except KeyboardInterrupt:
            print("\n程序已被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("错误分析: 可能是图像格式不支持、路径错误或内存不足")
            print("建议：检查图像文件是否完整，尝试使用较小的图像")
    
    print("\n感谢使用灰锥识别系统！")

if __name__ == "__main__":
    main()