from flask import Flask, render_template, request, jsonify, make_response
import json
import ssl

# import static.assets.py.IoT as IoT
import static.assets.py.Recognizer as Recognizer
import static.assets.py.mediapipe_gesture as mediapipe_gesture

import static.cursor.single as single

app = Flask(__name__)
#__name__ 인자는 정적파일과 템플릿을 찾는 데 쓰임 
import crawling
# iot = IoT.IoT()
mp_gesture = mediapipe_gesture.mediapipe_gesture()

@app.route('/')
def hello():
    news_list, href, img = crawling.news_crawling()
    img_list = crawling.img_src(news_list)
    sum_list = crawling.summary(news_list)

    return render_template('index.html', news_list = news_list, href = href, img = img, img_list=img_list, len = len(news_list), sum = sum_list )
    

@app.route('/update_iot', methods=['GET'])
def update_iot_status():
    # iot_update_result = iot.get_initial_state()
    iot_update_result = {'iot': 'success'}

    response = app.response_class(
        response = json.dumps(iot_update_result),
        status = 200,
        mimetype = 'application/json'
    )

    return response

# 220905 제스처 활성화 응답 함수
@app.route('/gesture_recog', methods=['POST'])
def gesture_recognition():
    try:
        print("gesture function start")
        result_dict = single.single_function("static/cursor/gesture_train.csv")
    except Exception as e:
        result_dict = {"gesture": "fail"}
        print(e)
        
    print(result_dict)

    response_dict = app.response_class(
            response = json.dumps(result_dict),
            status = 200,
            mimetype = 'application/json'
        )
    
    return response_dict

@app.route('/speech_recog', methods=['POST'])
def speech_recognition():
    speech_recog_result = request.get_json()
    user_said = speech_recog_result['speech_recog_result'][0]
    
    response_dict = {}

    recognizer = Recognizer.Recognizer(threshold=0.6)
    command = recognizer.what_user_said(user_said)
    print(command)

    # if command != 'no match':
    #     iot.command = command
    #     iot.socket.close()
    #     control_result = iot.control_iot()
    #     print(control_result)

    #     response_dict = app.response_class(
    #         response = json.dumps(control_result),
    #         status = 200,
    #         mimetype = 'application/json'
    #     )

    control_result = {'speech': 'success'}

    response_dict = app.response_class(
        response = json.dumps(control_result),
        status = 200,
        mimetype = 'application/json'
    )

    print('-'*30)
    return response_dict

if __name__ == '__main__':
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    # ssl_context.load_cert_chain(certfile='ssl/server.crt', keyfile='ssl/server.key', password='3680')
    
    # app.run(debug=True, host='0.0.0.0', port=3680, ssl_context=ssl_context)
    app.run(debug=True)
