from flask import Flask, render_template, request
import urllib.request
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from flask import jsonify
import uuid
from scipy import signal
import json
import pandas as pd

load_dotenv()

#Create the FLASK app
app = Flask(__name__)
model = load_model('3_CNN_PD.h5')
target_img = os.path.join(os.getcwd() ,'')




@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


allowed_device = 'device_allowed_123'  # Dispositivo permitido

ALLOWED_EXT1 = set(['csv'])
def allowed_file1(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT1
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    return img
def freq_get(data):

    t=data['miliseg'].values
    signal_x_ac = data['AcX'].values
    signal_y_ac = data['AcY'].values
    signal_z_ac = data['AcZ'].values
    signal_x_gy = data['GyX'].values
    signal_y_gy = data['GyY'].values
    signal_z_gy = data['GyZ'].values

    x_ac=signal_x_ac
    y_ac=signal_y_ac
    z_ac=signal_z_ac

    x_gy=signal_x_gy
    y_gy=signal_y_gy
    z_gy=signal_z_gy


#Eliminar pico en 0 

    x_ac_mean=np.mean(x_ac)
    y_ac_mean=np.mean(y_ac)
    z_ac_mean=np.mean(z_ac)

    x_gy_mean=np.mean(x_gy)
    y_gy_mean=np.mean(y_gy)
    z_gy_mean=np.mean(z_gy)

    x_total_ac=x_ac-x_ac_mean
    y_total_ac=y_ac-y_ac_mean
    z_total_ac=z_ac-z_ac_mean

    x_total_gy=x_gy-x_gy_mean
    y_total_gy=y_gy-y_gy_mean
    z_total_gy=z_gy-z_gy_mean

    frequencies = [3.25,12]  # Frecuencias en Hz

# Diseñar un filtro de banda
    b, a = signal.butter(2, frequencies, btype='band', fs=130, output='ba', analog=False)

# Aplicar el filtro a la señal IMU

    x_ac_filtered = signal.lfilter(b, a, x_total_ac)
    y_ac_filtered = signal.lfilter(b, a, y_total_ac)
    z_ac_filtered = signal.lfilter(b, a, z_total_ac)

    x_gy_filtered = signal.lfilter(b, a, x_total_gy)
    y_gy_filtered = signal.lfilter(b, a, y_total_gy)
    z_gy_filtered = signal.lfilter(b, a, z_total_gy)

    frequencies_x_ac, power_spectrum_x_ac = signal.periodogram(x_ac_filtered, 110,detrend=False)
    frequencies_y_ac, power_spectrum_y_ac = signal.periodogram(y_ac_filtered, 110,detrend=False)
    frequencies_z_ac, power_spectrum_z_ac = signal.periodogram(z_ac_filtered, 110,detrend=False)
    frequencies_x_gy, power_spectrum_x_gy = signal.periodogram(x_gy_filtered, 110,detrend=False)
    frequencies_y_gy, power_spectrum_y_gy = signal.periodogram(y_gy_filtered, 110,detrend=False)
    frequencies_z_gy, power_spectrum_z_gy = signal.periodogram(z_gy_filtered, 110,detrend=False)

# Obtener el índice de la frecuencia dominante
    dominant_frequency_index_x_ac = np.argmax(power_spectrum_x_ac)
    dominant_frequency_index_y_ac = np.argmax(power_spectrum_y_ac)
    dominant_frequency_index_z_ac = np.argmax(power_spectrum_z_ac)
    dominant_frequency_index_x_gy = np.argmax(power_spectrum_x_gy)
    dominant_frequency_index_y_gy = np.argmax(power_spectrum_y_gy)
    dominant_frequency_index_z_gy = np.argmax(power_spectrum_z_gy)

# Obtener el pico de potencia alrededor de la frecuencia dominante con ±0.3 Hz
    peak_power_x_ac = np.sum(power_spectrum_x_ac[dominant_frequency_index_x_ac-3:dominant_frequency_index_x_ac+3])
    peak_power_y_ac = np.sum(power_spectrum_y_ac[dominant_frequency_index_y_ac-3:dominant_frequency_index_y_ac+3])
    peak_power_z_ac = np.sum(power_spectrum_z_ac[dominant_frequency_index_z_ac-3:dominant_frequency_index_z_ac+3])
    peak_power_x_gy = np.sum(power_spectrum_x_gy[dominant_frequency_index_x_gy-3:dominant_frequency_index_x_gy+3])
    peak_power_y_gy = np.sum(power_spectrum_y_gy[dominant_frequency_index_y_gy-3:dominant_frequency_index_y_gy+3])
    peak_power_z_gy = np.sum(power_spectrum_z_gy[dominant_frequency_index_z_gy-3:dominant_frequency_index_z_gy+3])

    peaks_gyro=[peak_power_x_gy, peak_power_y_gy, peak_power_z_gy]

    sum_x_gy=np.sum(power_spectrum_x_gy)
    sum_y_gy=np.sum(power_spectrum_y_gy)
    sum_z_gy=np.sum(power_spectrum_z_gy)
    sumatorias=[sum_x_gy,sum_y_gy,sum_z_gy]

    max_peak_gyro = max(peaks_gyro)
    indice_maximo_gyro = peaks_gyro.index(max_peak_gyro)
    sumatoria=sumatorias[indice_maximo_gyro]

# Calcular threshold 
    thresh=max_peak_gyro/sumatoria

#Calcular peak power y freq
    peaks=[peak_power_x_ac,peak_power_y_ac,peak_power_z_ac,peak_power_x_gy, peak_power_y_gy, peak_power_z_gy]
    freqs=[frequencies_x_ac[dominant_frequency_index_x_ac],frequencies_y_ac[dominant_frequency_index_y_ac],frequencies_z_ac[dominant_frequency_index_z_ac],frequencies_x_gy[dominant_frequency_index_x_gy],frequencies_y_gy[dominant_frequency_index_y_gy],frequencies_z_gy[dominant_frequency_index_z_gy]]
    max_peak = max(peaks)
    indice_maximo = peaks.index(max_peak)
    freq=freqs[indice_maximo]

    datos = {'thresh':thresh,'peak': max_peak,'freq':freq}
    return jsonify(datos)
    
# Function to save freq images

@app.route('/manual',methods=['GET'])
def manual():
    return 'Aqui va el manual'

@app.route('/index.html',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/resultados',methods=['GET'])
def resultados():
    file_path2 = os.path.join('images','sample.jpg')
    db_filename = 'db.json'
    with open(db_filename, "r") as file:
        data = json.loads(file.read())
    l=list(filter(lambda x:x["id"]=="3d9cc8ed-bcf9-4c58-8080-fc84f41c25b8",data))
    url=l[0]['url']
    urllib.request.urlretrieve(url, 'static/images/sample.jpg')
    #return render_template('resultados.html', user_image = file_path2)
    return render_template('resultados.html')


@app.route('/txt',methods=['GET','POST'])
def txt():
    if request.method == 'POST':
        txt  = request.files('txt')
        if txt:
            print(txt)
            return 'not found'
        
@app.route('/check',methods=['GET','POST'])
def check():
    if request.method == 'POST':
        password  = request.form.get('pass')
        if password:
            print(password)
            db_filename = 'db.json'
            with open(db_filename, "r") as file:
                data = json.loads(file.read())
            for i in data:
                if password == i['id']:
                    return 'found'
            return 'not found'

@app.route('/register',methods=['GET','POST'])
def register():
    dic={}
    db_filename = 'db.json'
    with open(db_filename, "r") as file:
        data = json.loads(file.read())
    if request.method == 'POST':
        id  = request.form.get('id')
        name  = request.form.get('name')
        ape  = request.form.get('last name')
        ci  = request.form.get('ci')
        date  = request.form.get('date')  
        if id:
            dic['id']=id
        if name:
            dic['name']=name
        if ape:
            dic['last name']=ape
        if ci:
            dic['ci']=ci
        if date:
            dic['date']=date
    print(dic)
    return jsonify(dic)

@app.route('/freq',methods=['GET','POST'])
def freqWave():
    device_id = request.headers.get('Device-ID')
    if device_id == allowed_device:
        app.logger.info('in upload route')
        if request.method == 'POST':
            file_to_upload  = request.files['csv']
            print(type(file_to_upload))
            if file_to_upload  and allowed_file1(file_to_upload .filename):
                app.logger.info('%s file_to_upload', file_to_upload)
                if file_to_upload:
                    file_to_upload.save('static/sensor_data/data.csv')
                    file_sensor = pd.read_csv("static/sensor_data/data.csv")
                    data=freq_get(file_sensor)
                    return data
    else:
        return {'message': 'NOT ALLOWED'}, 401

@app.route('/predict',methods=['GET','POST'])
def predict():
    device_id = request.headers.get('Device-ID')
    if device_id == allowed_device:
        app.logger.info('in upload route')
        cloudinary.config(cloud_name = os.getenv('CLOUD_NAME'), api_key=os.getenv('API_KEY'), api_secret=os.getenv('API_SECRET'))
        upload_result = None
        if request.method == 'POST':
            file_to_upload  = request.files['file']
            print(type(file_to_upload))
            if file_to_upload  and allowed_file(file_to_upload .filename):
                app.logger.info('%s file_to_upload', file_to_upload)
                if file_to_upload:

                    upload_result = cloudinary.uploader.upload(file_to_upload)
                    app.logger.info(upload_result)

                    #URL TO CLOUDINARY:
                    url=upload_result.get('url')

                    urllib.request.urlretrieve(url, 'static/images/sample.jpg')
                    file_path1 = os.path.join('static/images','sample.jpg')
                    img = read_image(file_path1) #prepressing method
                    prediction=model.predict(img).tolist()
                    clas="YEI"
                    porcentaje=0
                    print(prediction)
                    if prediction[0][0]> prediction[0][1]:
                        clas="HC"
                        porcentaje=prediction[0][0]
                    elif prediction[0][1]> prediction[0][0]:
                        clas = "PD"
                        porcentaje=prediction[0][1]
                    else:
                        clas="Unknown"
                    datos = {'class': clas,'porcentaje':porcentaje}
                    print(datos)
                    return jsonify(datos)
            else:
                return "Unable to read the file. Please check file extension"
    else:
                return {'message': 'NOT ALLOWED'}, 401
            
                d = {'id': id, 'url': url, 'pred': fruit}
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
                db_filename = 'db.json'
                with open(db_filename, "r") as file:
                    data = json.loads(file.read())
                data.append(d)
                with open(db_filename, "w") as file:
                    json.dump(data, file)

                #return render_template('predict.html', fruit = fruit,prob=prediction, user_image = file_path2)-
if __name__ == '__main__':
    port = os.environ.get("PORT",8000)
    app.run(host="0.0.0.0", port=port, debug=False)


