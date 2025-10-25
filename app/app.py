from flask import Flask, request, render_template_string
import mlflow
import pandas as pd
import os

# 1. 创建一个 Flask 应用实例
app = Flask(__name__)

# 2. 加载模型
# (我们使用版本号 "2" 来加载，这是我们在上一步中确定的)
model_uri = "models:/HousePricePredictor/2" 
model = None
try:
    # 找到 mlruns 文件夹的绝对路径
    project_root = os.path.dirname(os.path.dirname(__file__))
    local_registry = "file:" + os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(local_registry)
    
    model = mlflow.pyfunc.load_model(model_uri) 
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("请确保你已经在 MLflow UI ([http://12_7.0.0.1:5000]) 中,")
    print("并且 'HousePricePredictor' 的 'Version 2' 是存在的。")
    print("网站将运行，但预测功能会报错。")

# 3. 定义网页内容 (HTML 模板)
# --- (这里是第一个重要修改) ---
# 为什么？ 我们不再“硬编码” value="8.3"，
# 而是使用 'form_values.get(..., ...)' 来动态填充值。
# .get('MedInc', '8.3') 的意思是：
# “尝试获取 'MedInc' 的值，如果找不到，就使用默认值 '8.3'”
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>房价预测器</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; background-color: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        form { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        label { font-weight: bold; }
        input { width: 100%; padding: 8px; box-sizing: border-box; border-radius: 4px; border: 1px solid #ccc; }
        button { grid-column: span 2; padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; font-size: 1.2em; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>加州房价预测器</h1>
        <form action="/predict" method="POST">
            
            <label for="MedInc">收入中位数 (MedInc):</label>
            <input type="number" step="0.1" id="MedInc" name="MedInc" value="{{ form_values.get('MedInc', '8.3') }}">
                        
            <label for="HouseAge">房屋年龄 (HouseAge):</label>    
            <input type="number" step="1" id="HouseAge" name="HouseAge" value="{{ form_values.get('HouseAge', '41') }}">
                        
            <label for="AveRooms">平均房间数 (AveRooms):</label>
            <input type="number" step="0.1" id="AveRooms" name="AveRooms" value="{{ form_values.get('AveRooms', '7.0') }}">
                        
            <label for="AveBedrms">平均卧室数 (AveBedrms):</label>
            <input type="number" step="0.1" id="AveBedrms" name="AveBedrms" value="{{ form_values.get('AveBedrms', '1.0') }}">
                        
            <label for="Population">人口 (Population):</label>
            <input type="number" step="1" id="Population" name="Population" value="{{ form_values.get('Population', '322') }}">
                        
            <label for="AveOccup">平均入住人数 (AveOccup):</label>
            <input type="number" step="0.1" id="AveOccup" name="AveOccup" value="{{ form_values.get('AveOccup', '2.5') }}">
                                 
            <label for="Latitude">纬度 (Latitude):</label>
            <input type="number" step="0.01" id="Latitude" name="Latitude" value="{{ form_values.get('Latitude', '37.88') }}">
                        
            <label for="Longitude">经度 (Longitude):</label>
            <input type="number" step="0.01" id="Longitude" name="Longitude" value="{{ form_values.get('Longitude', '-122.23') }}">
                        
            <button type="submit">预测价格</button>
        </form>  
        
        <div class="result">{{ prediction_text }}</div>
    </div>
</body>
</html>
"""

# 4. 定义“路由”（Routes）
@app.route('/')
def home():
    """当用户访问网站主页 (http://.../) 时，显示 HTML 模板。"""
    # --- (这里是第二个重要修改) ---
    # 为什么？ 我们传递一个空的 form_values 字典，
    # 这样 .get(..., '默认值') 就会自动使用默认值。
    return render_template_string(HTML_TEMPLATE, prediction_text="", form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    """当用户点击“预测”按钮时 (提交一个 POST 请求到 /predict)，执行此函数。"""
    
    global model
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="错误：模型未加载。", form_values=request.form)

    try:
        # 1. 从表单中获取用户输入的数据
        features = [       
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]   
        
        # 2. 把数据转换成模型需要的格式
        feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        input_data = pd.DataFrame([features], columns=feature_names)
                
        # 3. 使用模型进行预测
        prediction = model.predict(input_data)
                
        # 4. 准备要显示的结果
        # --- (这里是第三个、也是最关键的修改 - 修复百亿 Bug) ---
        # 为什么？ 我们的模型已经预测了完整美元金额。
        # 我们“删除”了 * 100000 这个乘法。
        # 我们只保留格式化（:,.2f 表示“带逗号，保留两位小数”）。
        output = f"预测房价: ${prediction[0]:,.2f}"
            
    except Exception as e:
        output = f"预测出错: {e}"
        
    # 5. 把结果显示回网页上
    # --- (这里是第四个重要修改) ---
    # 为什么？ 我们把用户输入的值 (request.form)
    # 再传回给模板，这样输入框就能显示用户刚填的值了。
    return render_template_string(HTML_TEMPLATE, prediction_text=output, form_values=request.form)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)