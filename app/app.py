from flask import Flask, request, render_template_string
import mlflow
import pandas as pd
import os  # <-- 修复 E261：这里有两个空格
from dotenv import load_dotenv  # <-- 修复 E261：这里有两个空格

app = Flask(__name__)
load_dotenv()  # <-- 新增：加载 .env 文件

# --- (这是修复：使用 DagsHub) ---
mlflow.set_tracking_uri(os.getenv("DAGSHUB_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")
# --- (修复结束) ---

# --- (这部分不变，它现在会从 DagsHub 加载) ---
model_uri = "models:/HousePricePredictor/1"
model = None

try:
    model = mlflow.pyfunc.load_model(model_uri)
    # --- (这是修复：更新提示信息) ---
    print("模型从 DagsHub 加载成功！")
except Exception as e:
    # --- (这是修复：更新提示信息) ---
    print(f"从 DagsHub 加载模型失败: {e}")

# --- (HTML 模板... 这部分与你提供的 app.py 保持一致) ---
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

        #map {
            height: 300px;
            width: 100%;
            margin-bottom: 20px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }

        .modal {
            display: none;
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
        var latInput = document.getElementById('Latitude');
        var lngInput = document.getElementById('Longitude');
        var modal = document.getElementById('alertModal');
        var modalMsg = document.getElementById('alertMessage');

        var initialLat = parseFloat(latInput.value);
        var initialLng = parseFloat(lngInput.value);

        var map = L.map('map').setView([36.778, -119.417], 6);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org' +
                             '/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        var marker = L.marker([initialLat, initialLng]).addTo(map);

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
                const value = parseFloat(input.value);

                if (isNaN(value) || value % 1 !== 0) {
                    invalidFields.push(field.name);
                }
            }

            if (invalidFields.length > 0) {
                event.preventDefault();
                showAlert(invalidFields.join('、') + " 必须为整数。");
            }
        });

        function updateMarkerAndForm(latlng) {
            marker.setLatLng(latlng);
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

        var baseUrl = "https://raw.githubusercontent.com/johan/world.geo.json";
        var geoJsonUrl = baseUrl + "/master/countries/USA/CA.geo.json";

        var boundaryStyle = {
            "color": "#007bff",
            "weight": 3,
            "fillOpacity": 0.1,
            "fillColor": "#007bff"
        };

        fetch(geoJsonUrl)
            .then(res => res.json())
            .then(data => {
                var californiaLayer = L.geoJson(data, {
                    style: boundaryStyle
                }).addTo(map);

                map.fitBounds(californiaLayer.getBounds());

                californiaLayer.on('click', function(e) {
                    updateMarkerAndForm(e.latlng);
                    L.DomEvent.stopPropagation(e);
                });

                map.on('click', function(e) {
                    showAlert("请在加州高亮区域内选择一个点。");
                });
            })
            .catch(err => {
                console.error("无法加载加州边界:", err);
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
    # (这部分与你提供的 app.py 保持一致)
    return render_template_string(
        HTML_TEMPLATE,
        prediction_text="",
        form_values={}
    )


@app.route('/predict', methods=['POST'])
def predict():
    # (这部分与你提供的 app.py 保持一致)
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="错误：模型未加载。",
            form_values=request.form
        )

    try:
        medInc = round(float(request.form['MedInc']), 2)
        houseAge = round(float(request.form['HouseAge']), 2)

        try:
            f_aveRooms = float(request.form['AveRooms'])
            f_aveBedrms = float(request.form['AveBedrms'])
            f_population = float(request.form['Population'])
            f_aveOccup = float(request.form['AveOccup'])
        except ValueError:
            return render_template_string(
                HTML_TEMPLATE,
                prediction_text="错误：输入包含无效数字。",
                form_values=request.form
            )

        if (f_aveRooms % 1 != 0 or
                f_aveBedrms % 1 != 0 or
                f_population % 1 != 0 or
                f_aveOccup % 1 != 0):
            return render_template_string(
                HTML_TEMPLATE,
                prediction_text="错误：房间数、卧室数和人数必须为整数。",
                form_values=request.form
            )

        aveRooms = f_aveRooms
        aveBedrms = f_aveBedrms
        population = f_population
        aveOccup = f_aveOccup

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

        output = f"预测房价: ${prediction[0]:,.2f}"

    except Exception as e:
        output = f"预测出错: {e}"

    # (修复 W293: 移除了这里可能存在的带空格的空行)
    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=output,
        form_values=request.form
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
