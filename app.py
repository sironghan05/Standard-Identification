from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 禁用Flask的默认JSON排序
app.config['JSON_SORT_KEYS'] = False

def load_image_from_bytes(image_bytes):
    """从字节数据加载图像"""
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图像")
        return img
    except Exception as e:
        print(f"加载图像失败: {str(e)}")
        return None

def preprocess_image(image):
    """预处理图像：灰度化、高斯模糊、自适应阈值二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def find_largest_contour(binary_image):
    """查找二值图像中最大的轮廓"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 找出面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    # 过滤掉面积过小的轮廓
    if cv2.contourArea(largest_contour) < 500:
        return None
    return largest_contour

def approximate_triangle(contour):
    """将轮廓近似为三角形"""
    perimeter = cv2.arcLength(contour, True)
    # 使用不同的epsilon值尝试近似
    for epsilon in [0.02, 0.03, 0.04]:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 3:
            return approx
    
    # 尝试使用凸包
    convex_hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(convex_hull, True)
    hull_approx = cv2.approxPolyDP(convex_hull, 0.03 * hull_perimeter, True)
    if len(hull_approx) >= 3:
        # 取凸包的前3个点作为三角形
        return hull_approx[:3]
    
    return None

def calculate_triangle_properties(triangle):
    """计算三角形的高、底边和是否为等腰三角形"""
    if triangle is None or len(triangle) < 3:
        return None, None, False
    
    pts = triangle.reshape(-1, 2)
    
    # 计算边长
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])** 2)
    
    # 判断是否是手动选择的点集（特征是三维形状 [[[x1,y1]], [[x2,y2]], [[x3,y3]]]）
    is_manual_selection = len(triangle.shape) == 3 and triangle.shape[1] == 1
    
    # 对于手动选择的点，使用第一个点作为顶点，后两个点作为底边端点
    if is_manual_selection:
        apex = triangle[0][0]
        p1, p2 = triangle[1][0], triangle[2][0]
        print(f"[三角形属性] 手动选择顶点坐标: {apex}, 底边点1: {p1}, 底边点2: {p2}")
    else:
        # 自动检测的情况，找到y坐标最小的点作为顶点
        apex_idx = np.argmin(pts[:, 1])
        apex = pts[apex_idx]
        
        # 剩下的两个点作为底边的两个端点
        base_idxs = [i for i in range(3) if i != apex_idx]
        p1, p2 = pts[base_idxs[0]], pts[base_idxs[1]]
    
    # 底边长度
    base = distance(p1, p2)
    
    # 计算高（顶点到底边的垂直距离）
    numerator = abs((p2[0] - p1[0]) * (apex[1] - p1[1]) - (p2[1] - p1[1]) * (apex[0] - p1[0]))
    denominator = base
    
    if denominator > 0:
        height = numerator / denominator
    else:
        height = 0
    
    # 检查是否是等腰三角形
    dist_to_p1 = distance(apex, p1)
    dist_to_p2 = distance(apex, p2)
    is_isosceles = abs(dist_to_p1 - dist_to_p2) < 0.1 * max(dist_to_p1, dist_to_p2)
    
    # 添加调试信息
    print(f"[三角形属性] 顶点坐标: {apex}, 底边点1: {p1}, 底边点2: {p2}")
    print(f"[三角形属性] 高度: {height:.2f}, 底边: {base:.2f}, 比例: {height/base:.3f}")
    
    return height, base, is_isosceles

def check_cone_requirements(height, base):
    """检查灰锥是否满足比例要求（高度/底边）"""
    if height is None or base is None or height == 0 or base == 0:
        return False, "无法识别三角形或计算比例"
    
    # 计算实际比例（高度/底边）
    actual_ratio = height / base
    min_ratio = 2.5  # 最小比例要求
    max_ratio = 3.3  # 最大比例要求
    
    # 检查比例是否在要求范围内
    is_valid = min_ratio <= actual_ratio <= max_ratio
    
    # 详细调试信息
    print(f"[灰锥检测] 实际比例(高度/底边): {actual_ratio:.3f}")
    print(f"[灰锥检测] 高度: {height:.2f}, 底边: {base:.2f}")
    print(f"[灰锥检测] 要求比例范围(高度/底边): {min_ratio}-{max_ratio}")
    print(f"[灰锥检测] 判定结果: {'符合要求' if is_valid else '不符合要求'}")
    
    # 根据判定结果返回正确信息
    if is_valid:
        return bool(is_valid), f"灰锥识别成功！实际比例(高度/底边): {actual_ratio:.2f}, 要求比例范围: {min_ratio}-{max_ratio}"
    else:
        return bool(is_valid), f"灰锥比例不符合要求。实际比例(高度/底边): {actual_ratio:.2f}, 要求比例范围: {min_ratio}-{max_ratio}"

def image_to_base64(image):
    """将OpenCV图像转换为base64编码"""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def draw_triangle_on_image(image, triangle, color=(0, 255, 0), thickness=2):
    """在图像上绘制三角形"""
    if triangle is not None:
        cv2.drawContours(image, [triangle], -1, color, thickness)
    return image

def draw_contour_on_image(image, contour, color=(255, 0, 0), thickness=2):
    """在图像上绘制轮廓"""
    if contour is not None:
        cv2.drawContours(image, [contour], -1, color, thickness)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 读取文件内容
        image_bytes = file.read()
        
        # 加载图像
        image = load_image_from_bytes(image_bytes)
        if image is None:
            return jsonify({'error': '无法加载图像'}), 400
        
        # 预处理
        binary = preprocess_image(image)
        
        # 查找轮廓
        contour = find_largest_contour(binary)
        
        # 初始结果
        result = {
            'auto_detect': {
                'success': False,
                'message': '无法识别灰锥',
                'height': None,
                'base': None,
                'ratio': None,
                'is_isosceles': False
            },
            'images': {
                'original': image_to_base64(image),
                'binary': image_to_base64(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
                'result': ''
            }
        }
        
        # 自动检测逻辑
        if contour is not None:
            # 尝试三角形近似
            triangle = approximate_triangle(contour)
            
            if triangle is not None:
                # 计算三角形属性
                height, base, is_isosceles = calculate_triangle_properties(triangle)
                
                if height is not None and base is not None:
                    # 检查是否符合灰锥比例要求
                    valid, message = check_cone_requirements(height, base)
                    
                    # 添加等腰三角形信息
                    triangle_info = " (近似等腰三角形)" if is_isosceles else " (非等腰三角形)"
                    message += triangle_info
                    
                    # 更新结果，确保所有值都能被JSON序列化
                    result['auto_detect'] = {
                        'success': bool(valid),  # 确保是Python原生布尔值
                        'message': message,
                        'height': float(round(height, 2)) if height is not None else None,  # 确保是浮点数
                        'base': float(round(base, 2)) if base is not None else None,  # 确保是浮点数
                        'ratio': float(round(height/base, 3)) if height is not None and base is not None and base > 0 else None,  # 确保是浮点数
                        'is_isosceles': bool(is_isosceles)  # 确保是Python原生布尔值
                    }
                    
                    # 绘制结果图像
                    result_image = image.copy()
                    result_image = draw_contour_on_image(result_image, contour)
                    result_image = draw_triangle_on_image(result_image, triangle)
                    result['images']['result'] = image_to_base64(result_image)
                else:
                    # 只有轮廓没有三角形的情况
                    result_image = image.copy()
                    result_image = draw_contour_on_image(result_image, contour)
                    result['images']['result'] = image_to_base64(result_image)
            else:
                # 只有轮廓没有三角形的情况
                result_image = image.copy()
                result_image = draw_contour_on_image(result_image, contour)
                result['images']['result'] = image_to_base64(result_image)
        else:
            # 没有检测到轮廓的情况
            result['images']['result'] = image_to_base64(image)
        
        return jsonify(result)
    except Exception as e:
        print(f"检测过程中发生错误: {str(e)}")
        return jsonify({'error': f'检测过程中发生错误: {str(e)}'}), 500

@app.route('/manual-detect', methods=['POST'])
def manual_detect():
    try:
        # 获取请求数据
        data = request.get_json()
        image_data = data.get('image_data')
        points = data.get('points')
        
        if not image_data or not points or len(points) != 3:
            return jsonify({'error': '无效的请求数据'}), 400
        
        # 从base64解码图像
        header, image_str = image_data.split(',', 1)
        image_bytes = base64.b64decode(image_str)
        image = load_image_from_bytes(image_bytes)
        
        if image is None:
            return jsonify({'error': '无法加载图像'}), 400
        
        # 转换手动选择的点为三角形
        user_triangle = np.array([[[int(p[0]), int(p[1])]] for p in points], dtype=np.int32)
        
        # 计算三角形属性
        height, base, is_isosceles = calculate_triangle_properties(user_triangle)
        
        # 检查是否符合灰锥比例要求
        valid, message = check_cone_requirements(height, base)
        
        # 添加等腰三角形信息
        triangle_info = " (近似等腰三角形)" if is_isosceles else " (非等腰三角形)"
        message += triangle_info
        
        # 绘制结果图像
        result_image = image.copy()
        
        # 绘制三角形
        for i in range(3):
            cv2.circle(result_image, tuple(points[i]), 5, (0, 255, 0), -1)
            cv2.line(result_image, tuple(points[i]), tuple(points[(i+1)%3]), (255, 0, 0), 2)
        
        # 构建结果，确保所有值都能被JSON序列化
        result = {
            'success': bool(valid),  # 确保是Python原生布尔值
            'message': message,
            'height': float(round(height, 2)) if height is not None else None,  # 确保是浮点数
            'base': float(round(base, 2)) if base is not None else None,  # 确保是浮点数
            'ratio': float(round(height/base, 3)) if height is not None and base is not None and base > 0 else None,  # 确保是浮点数
            'is_isosceles': bool(is_isosceles),  # 确保是Python原生布尔值
            'image': image_to_base64(result_image)
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"手动检测过程中发生错误: {str(e)}")
        return jsonify({'error': f'手动检测过程中发生错误: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)