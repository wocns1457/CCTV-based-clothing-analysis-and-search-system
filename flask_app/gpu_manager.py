import os
import time
import requests

from threading import Thread

from googleapiclient import discovery
from google.auth import default

from flask import Flask, request, jsonify, Response, stream_with_context


app = Flask(__name__)

# GCP 프로젝트 및 인스턴스 정보
PROJECT = " "
ZONE = " "
INSTANCE_NAME = " "

GPU_SERVER = " "  # GPU 서버의 내부 IP 

credentials, _ = default()
compute = discovery.build('compute', 'v1', credentials=credentials)

last_request_time = time.time()
INACTIVITY_LIMIT = 1800 # 30분 (초 단위)

def get_instance_status():
    return compute.instances().get(project=PROJECT, zone=ZONE, instance=INSTANCE_NAME).execute()['status']

def start_instance():
    """GCP 인스턴스 시작"""
    print("Starting GPU instance...")
    compute.instances().start(project=PROJECT, zone=ZONE, instance=INSTANCE_NAME).execute()
    # 인스턴스가 활성화될 때까지 대기
    while True:
        if get_instance_status() == 'RUNNING':
            print("Instance is running.")
            break
        time.sleep(2)

def stop_instance():
    """GCP 인스턴스 중지"""
    print("Stopping GPU instance...")
    compute.instances().stop(project=PROJECT, zone=ZONE, instance=INSTANCE_NAME).execute()

def inactivity_monitor():
    """활성 상태 모니터링"""
    global last_request_time
    while True:
        time.sleep(600)  # 10분마다 확인
        if time.time() - last_request_time > INACTIVITY_LIMIT:
            print("No requests for a while. Stopping the instance.")
            stop_instance()

# 비활성화 모니터링 스레드 실행
Thread(target=inactivity_monitor, daemon=True).start()


@app.route('/process_request', methods=['POST'])
def process_request():
    """클라이언트 요청 처리"""
    global last_request_time
    try:
        # 마지막 요청 시간 업데이트
        last_request_time = time.time()

        # GCP 인스턴스 시작
        if get_instance_status() != 'RUNNING':
            start_instance()
            # GPU서버 오픈 지연 시간
            time.sleep(65)

        while True:
            try:
                response = requests.get(GPU_SERVER)
            except:
                time.sleep(5)
                response = requests.get(GPU_SERVER)
            if response.status_code==200:
                break

        print('GPU flask server is running...')

         # 클라이언트 요청 데이터 전달
        response = requests.post(GPU_SERVER + '/predict_video', files=request.files, data=request.form, stream=True)

         # 스트리밍 응답을 클라이언트로 전달
        def generate():
             for line in response.iter_lines():
                 if line:
                     yield line.decode('utf-8') + '\n'
        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
