from flask import Flask, request, render_template_string
import mlflow
import pandas as pd

app = Flask(__name__)

# --- (修复：使用相对路径) ---
mlflow.set_tracking_uri("file:./mlruns")

# --- (修复：按版本号加载) ---
model_uri = "models:/HousePricePredictor/1"
model = None

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型失败: {e}")

# --- (这是更新后的 HTML 模板) ---
# (修复 E501: 对过长的行进行了换行)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>房价预测器</title>

    <link rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""/>

    <style>
        body { font-family: Arial, sans-serif; margin: 2em;
               background-color: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: #fff;
                     padding: 20px; border-radius: 8px;
                     box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        form { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        label { font-weight: bold; }
        input { width: 100%; padding: 8px; box-sizing: border-box;
                border-radius: 4px; border: 1px solid #ccc; }
        button { grid-column: span 2; padding: 10px;
                 background-color: #007bff; color: white; border: none;
                 border-radius: 4px; cursor: pointer; font-size: 16px; }
        .result { margin-top: 20px; font-size: 1.2em; font-weight: bold; }

        /* 2. 为地图容器设置一个大小 */
        #map {
            height: 300px;
            width: 100%;
            margin-bottom: 20px; /* 在地图和表单之间添加一些间距 */
            border-radius: 8px;
            border: 1px solid #ccc; /* 给地图加个边框 */
        }

        /* 3. 自定义弹窗 (替代 alert()) */
        .modal {
            display: none; /* 默认隐藏 */
            position: fixed; z-index: 1000; left: 0; top: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.4);
            transition: opacity 0.2s ease;
        }
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto; padding: 20px;
            border: 1px solid #888; border-radius: 8px;
            width: 80%; max-width: 300px; text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .modal-close {
            background-color: #007bff; color: white; padding: 8px 12px;
            border: none; border-radius: 4px; cursor: pointer;
            margin-top: 15px; font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>加州房价预测器</h1>

        <p>请点击地图**加州范围内**选择位置：</p>

        <div id="map"></div>

        <form action="/predict" method="POST">
            <label for="MedInc">收入中位数 (W):</label>
            <input type="number" step="0.1" id="MedInc" name="MedInc"
                   value="{{ form_values.get('MedInc', '8.3') }}">

            <label for="HouseAge">房屋年龄 (HouseAge):</label>
            <input type="number" step="1" id="HouseAge" name="HouseAge"
                   value="{{ form_values.get('HouseAge', '41') }}">

            <label for="AveRooms">希望房间数 (AveRooms):</label>
            <input type="number" step="1" id="AveRooms" name="AveRooms"
                   value="{{ form_values.get('AveRooms', '7') }}">

            <label for="AveBedrms">希望卧室数 (AveBedrms):</label>
            <input type="number" step="1" id="AveBedrms" name="AveBedrms"
                   value="{{ form_values.get('AveBedrms', '1') }}">

            <label for="Population">人口 (Population):</label>
            <input type="number" step="1" id="Population" name="Population"
                   value="{{ form_values.get('Population', '322') }}">

            <label for="AveOccup">入住人数 (AveOccup):</label>
            <input type="number" step="1" id="AveOccup" name="AveOccup"
                   value="{{ form_values.get('AveOccup', '3') }}">

            <label for="Latitude">纬度 (Latitude):</label>
            <input type="number" step="any" id="Latitude" name="Latitude"
                   value="{{ form_values.get('Latitude', '37.88') }}">

            <label for="Longitude">经度 (Longitude):</label>
            <input type="number" step="any" id="Longitude" name="Longitude"
                   value="{{ form_values.get('Longitude', '-122.23') }}">

            <button type="submit">预测价格</button>
        </form>

        <div class="result">{{ prediction_text }}</div>
    </div>

    <div id="alertModal" class="modal">
        <div class="modal-content">
            <p id="alertMessage"></p>
            <button class="modal-close" onclick="closeAlert()">好的</button>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
            crossorigin=""></script>

    <script>
        // --- 1. 获取表单和弹窗元素 ---
        var latInput = document.getElementById('Latitude');
        var lngInput = document.getElementById('Longitude');
        var modal = document.getElementById('alertModal');
        var modalMsg = document.getElementById('alertMessage');

        // --- (已删除) 删除了 roundUpOnInput 函数 ---

        // --- (已删除) 删除了 onchange 事件绑定 ---

        var initialLat = parseFloat(latInput.value);
        var initialLng = parseFloat(lngInput.value);

        // --- 2. 立即初始化地图 (保证地图总能显示) ---
        var map = L.map('map').setView([36.778, -119.417], 6);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org' +
                             '/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        var marker = L.marker([initialLat, initialLng]).addTo(map);

        // --- (新增) 修复问题3: 表单提交前验证整数 ---
        var form = document.querySelector('form');
        form.addEventListener('submit', function(event) {
            const fieldsToCheck = [
                { id: 'AveRooms', name: '希望房间数' },
                { id: 'AveBedrms', name: '希望卧室数' },
                { id: 'Population', name: '人口' },
                { id: 'AveOccup', name: '入住人数' }
            ];

            let invalidFields = [];
            for (const field of fieldsToCheck) {
                const input = document.getElementById(field.id);
                // 检查值是否为数字，以及是否 浮点数 % 1 不等于 0 (即有小数)
                const value = parseFloat(input.value);

                if (isNaN(value) || value % 1 !== 0) {
                    invalidFields.push(field.name);
                }
            }

            if (invalidFields.length > 0) {
                event.preventDefault(); // 阻止表单提交
                showAlert(invalidFields.join('、') + " 必须为整数。");
            }
        });
        // --- 修复结束 ---

        // --- 3. 辅助函数 ---
        function updateMarkerAndForm(latlng) {
            marker.setLatLng(latlng);
            // (修复问题2): JS 这边也放开精度限制，使用 toFixed(6) 提高精度
            latInput.value = latlng.lat.toFixed(6);
            lngInput.value = latlng.lng.toFixed(6);
        }
        function showAlert(message) {
            modalMsg.textContent = message;
            modal.style.display = "block";
        }
        function closeAlert() {
            modal.style.display = "none";
        }
        window.onclick = function(event) {
            if (event.target == modal) {
                closeAlert();
            }
        }

        // --- 4. 尝试加载加州边界 (这部分逻辑在您的代码中已正确) ---

        // *** (修复问题1): 更换为更可靠的 GeoJSON URL ***
        // (修复 E501: 将过长的 URL 字符串拆分为两行)
        var baseUrl = "https://raw.githubusercontent.com/johan/world.geo.json";
        var geoJsonUrl = baseUrl + "/master/countries/USA/CA.geo.json";

        var boundaryStyle = {
            "color": "#007bff",      // 边框颜色 (蓝色)
            "weight": 3,             // 边框宽度
            "fillOpacity": 0.1,      // 填充透明度
            "fillColor": "#007bff"    // 填充颜色
        };

        fetch(geoJsonUrl)
            .then(res => res.json())
            .then(data => {
                // --- 5.1 加载成功 (问题1的实现) ---
                var californiaLayer = L.geoJson(data, {
                    style: boundaryStyle
                }).addTo(map);

                map.fitBounds(californiaLayer.getBounds()); // 自动缩放

                // 核心：点击事件只绑在加州图层上
                californiaLayer.on('click', function(e) {
                    updateMarkerAndForm(e.latlng);
                    L.DomEvent.stopPropagation(e); // 阻止事件冒泡到 map
                });

                // 如果点击了加州以外 (即地图本身)
                map.on('click', function(e) {
                    showAlert("请在加州高亮区域内选择一个点。");
                });
            })
            .catch(err => {
                // --- 5.2 加载失败 (网络问题等) ---
                console.error("无法加载加州边界:", err);
                // 优雅降级：回退到“简单模式”
                map.on('click', function(e) {
                    updateMarkerAndForm(e.latlng);
                });
            });
    </script>

</body>
</html>
"""


@app.route('/')
def home():
    # 模板现在更丰富了，但我们传参的方式不变
    # (修复 E11x, E12x: 修正参数缩进)
    return render_template_string(
        HTML_TEMPLATE,
        prediction_text="",
        form_values={}
    )


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # (修复 E11x, E12x, E501: 修正参数缩进)
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="错误：模型未加载。",
            form_values=request.form
        )

    try:
        # --- 5. 后端数据清理和验证 (根据你的新要求) ---

        # 收入和房龄保持不变 (四舍五入到2位小数)
        medInc = round(float(request.form['MedInc']), 2)
        houseAge = round(float(request.form['HouseAge']), 2)

        # *** (修复问题3): 后端验证整数 ***
        try:
            f_aveRooms = float(request.form['AveRooms'])
            f_aveBedrms = float(request.form['AveBedrms'])
            f_population = float(request.form['Population'])
            f_aveOccup = float(request.form['AveOccup'])
        except ValueError:
            # (修复 E11x, E12x: 修正参数缩进)
            return render_template_string(
                HTML_TEMPLATE,
                prediction_text="错误：输入包含无效数字。",
                form_values=request.form
            )

        # 验证检查
        if (f_aveRooms % 1 != 0 or
                f_aveBedrms % 1 != 0 or
                f_population % 1 != 0 or
                f_aveOccup % 1 != 0):
            # (修复 E11x, E12x: 修正参数缩进)
            return render_template_string(
                HTML_TEMPLATE,
                prediction_text="错误：房间数、卧室数和人数必须为整数。",
                form_values=request.form
            )

        # 验证通过，使用这些值
        aveRooms = f_aveRooms
        aveBedrms = f_aveBedrms
        population = f_population
        aveOccup = f_aveOccup
        # *** 修复结束 ***
        # (修复 E303: 移除了这里多余的空行)
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])

        features = [
            medInc,
            houseAge,
            aveRooms,
            aveBedrms,
            population,
            aveOccup,
            latitude,
            longitude
        ]

        feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]

        input_data = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_data)

        # --- (修复：删除 * 100000) ---
        output = f"预测房价: ${prediction[0]:,.2f}"

    except Exception as e:
        output = f"预测出错: {e}"

    # 预测后，我们仍然返回模板，它会保留用户最后点击的经纬度值
    # (修复 E11x, E12x: 修正参数缩进)
    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=output,
        form_values=request.form
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
