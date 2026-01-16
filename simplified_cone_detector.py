import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_image(image_path):
    """加载图像并进行基本检查"""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
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

def put_chinese_text(img, text, position, size=20, color=(0, 255, 0)):
    """使用PIL在OpenCV图像上显示中文"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("simhei.ttf", size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=tuple(reversed(color)))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_user_selected_vertices(image):
    """让用户手动点击图像中的三个顶点"""
    print("请在图像上点击灰锥的三个顶点（建议顺序：顶部、左下、右下）")
    
    display_image = image.copy()
    clicked_points = []
    
    # 初始中文提示 - 使用正确的方式设置初始显示
    temp_img = image.copy()
    temp_img = put_chinese_text(temp_img, "请点击三点", (10, 30), size=24)
    display_image[:] = temp_img
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 3:
            clicked_points.append((x, y))
            
            # 每次都从原始图像开始，避免文字重叠
            temp_img = image.copy()
            
            # 绘制所有已选的点
            for point in clicked_points:
                cv2.circle(temp_img, point, 5, (0, 255, 0), -1)
            
            # 添加当前进度文本
            temp_img = put_chinese_text(temp_img, f"已选 {len(clicked_points)}/3 点", (10, 30), size=24)
            
            # 绘制连线
            if len(clicked_points) > 1:
                for i in range(len(clicked_points) - 1):
                    cv2.line(temp_img, clicked_points[i], clicked_points[i+1], (255, 0, 0), 2)
                if len(clicked_points) == 3:
                    cv2.line(temp_img, clicked_points[2], clicked_points[0], (255, 0, 0), 2)
            
            # 更新display_image
            display_image[:] = temp_img
            cv2.imshow("选择灰锥顶点", display_image)
    
    cv2.namedWindow("选择灰锥顶点", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("选择灰锥顶点", mouse_callback)
    cv2.imshow("选择灰锥顶点", display_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            cv2.destroyWindow("选择灰锥顶点")
            return None
        
        if len(clicked_points) == 3:
            # 重新复制原始图像，只添加确认提示，避免文字重叠
            temp_img = image.copy()
            # 重新绘制已选的点和线
            for point in clicked_points:
                cv2.circle(temp_img, point, 5, (0, 255, 0), -1)
            # 绘制连线
            cv2.line(temp_img, clicked_points[0], clicked_points[1], (255, 0, 0), 2)
            cv2.line(temp_img, clicked_points[1], clicked_points[2], (255, 0, 0), 2)
            cv2.line(temp_img, clicked_points[2], clicked_points[0], (255, 0, 0), 2)
            # 添加新的提示文本
            temp_img = put_chinese_text(temp_img, "Enter确认, ESC重选", (10, 30), size=24, color=(0, 255, 255))
            cv2.imshow("选择灰锥顶点", temp_img)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter键
                    cv2.destroyWindow("选择灰锥顶点")
                    return np.array([[[x, y]] for x, y in clicked_points], dtype=np.int32)
                elif key == 27:
                    clicked_points = []
                    display_image = image.copy()
                    display_image = put_chinese_text(display_image, "请点击三点", (10, 30), size=24)
                    cv2.imshow("选择灰锥顶点", display_image)
                    break

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
        return True, f"灰锥识别成功！实际比例(高度/底边): {actual_ratio:.2f}, 要求比例范围: {min_ratio}-{max_ratio}"
    else:
        return False, f"灰锥比例不符合要求。实际比例(高度/底边): {actual_ratio:.2f}, 要求比例范围: {min_ratio}-{max_ratio}"

def visualize_result(image, binary, contour, triangle, result_text):
    """可视化检测结果"""
    plt.figure(figsize=(12, 8))
    
    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 二值化图像
    plt.subplot(222)
    plt.imshow(binary, cmap='gray')
    plt.title('二值化图像')
    plt.axis('off')
    
    # 检测结果
    plt.subplot(223)
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

def detect_cone(image_path):
    """精简版灰锥检测函数"""
    print(f"\n=== 正在处理图像: {image_path} ===")
    
    # 加载图像
    image = load_image(image_path)
    if image is None:
        return False, "无法加载图像"
    
    # 预处理
    binary = preprocess_image(image)
    
    # 查找轮廓
    contour = find_largest_contour(binary)
    
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
                # 确保三角形信息添加在消息末尾
                message += triangle_info
                
                # 显示结果
                print(f"自动检测结果: {message}")
                visualize_result(image, binary, contour, triangle, message)
                
                # 询问是否手动修正
                print("是否需要手动选择顶点进行修正？(y/n)")
                if input().strip().lower() == 'y':
                    print("启动手动顶点选择...")
                    user_triangle = get_user_selected_vertices(image)
                    if user_triangle is not None:
                        # 正确计算三角形属性并检查比例要求
                        u_height, u_base, u_isosceles = calculate_triangle_properties(user_triangle)
                        u_valid, u_message = check_cone_requirements(u_height, u_base)
                        u_triangle_info = " (近似等腰三角形)" if u_isosceles else " (非等腰三角形)"
                        # 确保三角形信息添加在消息末尾
                        u_message += u_triangle_info
                        
                        print(f"手动选择检测结果: {u_message}")
                        visualize_result(image, binary, contour, user_triangle, u_message)
                        return u_valid, f"手动选择顶点: {u_message}"
                
                return valid, message
    
    # 如果自动检测失败，尝试手动选择
    print("自动检测失败，是否要手动选择顶点？(y/n)")
    if input().strip().lower() == 'y':
        user_triangle = get_user_selected_vertices(image)
        if user_triangle is not None:
            # 正确计算三角形属性并检查比例要求
            u_height, u_base, u_isosceles = calculate_triangle_properties(user_triangle)
            u_valid, u_message = check_cone_requirements(u_height, u_base)
            u_triangle_info = " (近似等腰三角形)" if u_isosceles else " (非等腰三角形)"
            # 确保三角形信息添加在消息末尾
            u_message += u_triangle_info
            
            print(f"手动选择检测结果: {u_message}")
            visualize_result(image, binary, contour, user_triangle, u_message)
            return u_valid, f"手动选择顶点: {u_message}"
    
    # 显示最终结果
    visualize_result(image, binary, contour, None, "无法识别灰锥")
    return False, "无法识别灰锥"

def main():
    """主函数"""
    print("===== 灰锥识别系统（精简版）=====")
    print("系统说明：使用自适应阈值方法进行灰锥检测，支持手动修正")
    print("\n拍摄建议：")
    print("1. 在光线均匀的环境下拍摄")
    print("2. 确保灰锥占据图像中心位置")
    print("3. 避免反光和阴影")
    print("4. 保持背景简洁")
    print("\n按Ctrl+C可以随时退出程序")
    
    while True:
        try:
            image_path = input("\n请输入图像路径 (或直接按Enter使用默认测试图像): ")
            
            if not image_path.strip():
                image_path = "测试图片\\gray_cone.png"
                print(f"使用默认图像路径: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在。请检查路径是否正确: {image_path}")
                continue
            
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in valid_extensions:
                print(f"错误: 不支持的文件格式 '{ext}'")
                continue
            
            # 执行检测
            valid, message = detect_cone(image_path)
            
            # 显示结果
            print("\n=== 检测结果 ===")
            print(f"{'✓' if valid else '✗'} {message}")
            
        except KeyboardInterrupt:
            print("\n程序已退出")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()