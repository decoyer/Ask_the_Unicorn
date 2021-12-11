from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import MobileNetV2
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import os, re, googletrans
import numpy as np

# 업로드 경로 지정
UPLOAD_FOLDER = 'static/images'

app = Flask(__name__)
app.secret_key = "flask"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 구글 번역 객체 생성
translator = googletrans.Translator()

def predict(filename):
    # 모델 객체 생성
    model = MobileNetV2(weights='imagenet')
    
    # 이미지 전처리
    img = load_img('static/images/' + filename, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # 이미지 예측
    y_pred = model.predict(x)
    label = decode_predictions(y_pred)
    label = label[0][0]

    # 예측 결과 처리
    result = label[1]
    result = re.sub('\_|\.', ' ', result)
    result = translator.translate(result, dest='ko')
    result = result.text

    # 정확도 계산
    acc = label[2] * 100
    
    return result, acc

# 메인 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 예측
@app.route('/', methods=['POST'])
def act():
    if request.method == 'POST':
        # 업로드 파일 처리
        file = request.files['file']

        if file:
            # 이미지 보안 처리
            x = secure_filename(file.filename)
            # 이미지 저장
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], x))
            # 이미지 예측
            predict(x)
            label, acc = predict(x)
            # 예측 결과 반환
            flash('"%.2f%%의 확률로 %s입니다."' % (acc, label))
            
            return render_template('index.html', image_file='/images/'+file.filename)

if __name__ == "__main__":
    # Flask 실행
    app.run(host='localhost', port='5000', debug=False)
