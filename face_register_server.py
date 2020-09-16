#!/usr/bin/python
import os
import sys
import socket
import cv2
import numpy
import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import base64
from io import BytesIO
import face_recognition
import datetime  # 날짜를 받아오기 위해 쓰는 라이브러리
import pickle
from datetime import datetime, timedelta
import time
import select

known_face_encodings = []
known_face_metadata = []


def restart():
    print("얼굴 등록이 완료되었습니다.")
    executable = sys.executable
    args = sys.argv[:]
    args.insert(0, sys.executable)

    time.sleep(1)
    print("Respawning...")
    os.execvp(executable, args)


# socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def save_known_faces():
    with open("known_faces.dat",
              "wb") as face_data_file:  # 파이썬 내장함수 open은 파일이름과 파일 열기모드를 입력값으로 받고 결과값으로 파일 객체를 돌려준다. known_faces라는 dat
        # 파일을 만든다. 이를 face_data_file로 칭한다.
        face_data = [known_face_encodings,
                     known_face_metadata]  # face_data는 known_face_encodings, known_face_metadata를 갖는 리스트이다.
        pickle.dump(face_data, face_data_file)  # 위에서 만든 리스트 face_data를 face_data_file(known_faces.dat)에 저장한다.
        print("Known faces backed up to disk.")  # 파일이 새로 초기화 되었음을 알린다.


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:  # known_faces.dat을 열어 face_data_file이라고 지칭한다.
            known_face_encodings, known_face_metadata = pickle.load(
                face_data_file)  # known_face_encodings, known_face_metadata 리스트를 face_data_file로 로드
            print("Known faces loaded from disk.")  # disk로부터 로드한 것을 출력
    except FileNotFoundError as e:  # 파일이 없으면 에러 문구를 발생하도록 한다.
        print("No previous face data found - starting with a blank known face list.")  # 파일이 발견되지 않았습니다.
        pass


def register_new_face(face_encoding, face_image):
    known_face_encodings.append(face_encoding)  # 리스트의 맨 마지막에 face_encoding을 더해준다.
    known_face_metadata.append({  # known_face_metadata 리스트에 아래와 같은 정보를 더해준다.
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
        "face_dist": 0,
    })



def lookup_known_face(face_encoding):
    metadata = None  # metadata 변수를 초기화

    if len(known_face_encodings) == 0:  # known_face_encodings 리스트가 비었다면, 즉 처음 인식되었다면.
        return metadata  # metadata를 리턴한다.

    # known_face 리스트 내에 있는 모든 얼굴과 unknown face 사이의 얼굴 거리(얼굴 유사성)을 계산한다.
    # 이 값은 0~1사이로 표현되며 숫자가 낮을 수록 얼굴이 비슷한 것을 나타낸다.
    face_distances = face_recognition.face_distance(known_face_encodings,
                                                    face_encoding)  # face_recognition.face_distance 가 가리키는 것이 바로 얼굴
    # 유사성 테스트이다.

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = numpy.argmin(face_distances)  # 리스트에서 비교한 값들 중에 최소값의 위치를 best_match_index 로 저장한다.

    # 얼굴 유사성이 0.6 이하이면 얼굴이 맞다고 판단한다.
    # 0.6이라는 값은 얼굴인식모델이 딥러닝 한 값을 나타낸다. 같은 사람일  경우 0.6보다 낮은 값을 나타내었다.
    # 유사성을 낮출수록 얼굴 인식을 더 깐깐하게 할 수 있다.
    if face_distances[best_match_index] < 0.52:  # 얼굴 유사성이 0.5보다 작을 경우
        # metadata 를 known_face_metadata 리스트의 best_match_index 자리의 값으로 둔다.
        metadata = known_face_metadata[best_match_index]

        # metadata 값에 기존에 저장되어 있던 last_seen과 seen_frames을 업데이트 한다.
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1
        metadata["face_dist"] = face_distances[best_match_index]

        # 얼굴에 비친 사람이 5분 이후에 비치게한다면 seen_count를 올려준다.
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=0.5):
            metadata["first_seen_this_interaction"] = datetime.now()
            metadata["seen_count"] += 100

    return metadata  # 그리고 어찌됐든 metadata를 리턴한다.


# 켜놓는 동안 파일을 tcp로부터 전송받은 파일을 계속해서 받아서 jpg로 저장했으면 좋겠다.
# 어차피 사용자 등록 모드와 경비모드 두가지를 만들 것이니까 이걸 어떻게 할지부터 생각하자.
# 등록모드에서는 .dat에 저장함으로써 거기서 등록이 완료되었다는 걸을 다시 front에 알려주는 역할이 어떤가


def main_loop():
    print('main loop')
    # Mysql 불러오기
    engine = create_engine('mysql+pymysql://root:1234qwer@localhost/IMAGE', echo=False)
    # 현재 시간으로 파일을 저장하는데 쓰이는 변수 nowDate
    nowDate = datetime.now()

    # 수신에 사용될 내 ip와 내 port번호
    TCP_IP = '164.125.234.91'
    TCP_PORT = 5004
    # TCP소켓 열고 수신 대기
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    # 접속을 기다리는 단계
    s.listen(True)
    # front가 접속되어 연결되었을 때 결과값이 리턴되는 함수
    conn, addr = s.accept()
    s.listen(True)

    while True:
        buffer = BytesIO()
        # 여기서 if문으로 만약 tcp로 데이터가 오지 않으면 다시 프로그램을 시작하도록 한다.
        # 추가적으로 else로 tcp를 통해 데이터가 도착하지 않았다는 것을 알리면 좋을 것 같다.
        number_of_faces_since_save = 0  # 얼마나 많은 알려진 얼굴을 디스크에서 백업했는지 알려준다.
        # String형의 이미지를 수신받아서 이미지로 변환 하고 화면에 출력
        length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
        if length is None:  # 보낸 데이터가 없으면 오류 구문을 나타낸다.
            print("클라이언트와 연결이 끊겼습니다.")
            cv2.destroyAllWindows()
            time.sleep(2)
            restart()
            break
        else:
            stringData = recvall(conn, int(length))
            data = numpy.fromstring(stringData, dtype='uint8')

            # 동영상 받기
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            cv2.imshow('Camera Showing', frame)

            # 빠른 얼굴인식을 위해 1/4사이즈로 줄인다.
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # BGR 컬러를 RGB 칼라로 변환시킨다. 이를 rgb_small_frame이라고 칭한다/home/jae.
            rgb_small_frame = small_frame[:, :, ::-1]

            # face_recognition 라이브러리의 face_loaction과 face_encoding을 통해 위치와 encoding을 찾는다.
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # 라벨 달아주기
            face_labels = []  # face_labels 라는 리스트 초기화

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # metadata는 lookup_known_face의 함수에 face_encoding값을 넣어서 받는다.
                metadata = lookup_known_face(face_encoding)
                # 이 metadata가 얼굴 인식한 사진이다.

                # 얼굴을 발견한다면 라벨링을 한다. 아래는 라벨링을 하게 되는 것이다.
                if metadata is not None:
                    time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                    face_label = f"At door {int(time_at_door.total_seconds())}s"  # 문 앞에 있는 시간을 출력하기 위한 라벨링 리스트 값을 넣는다.
                    if(int(time_at_door.total_seconds()>=20)):
                        conn.send('CC'.encode('utf-8'))  # 이부분을 conn으로 변경하였다.
                        save_known_faces()
                        s.close()
                        time.sleep(3)
                        restart()
                # 침입자 감지시 아래의 코드가 실행
                else:
                    # 여기서 new visitor면 client로 메세지 전송
                    face_label = "REGISTERING...!"
                    # 현재 전송받은 비디오에서 얼굴에 대한 이미지를 받는다.
                    top, right, bottom, left = face_location  # 위의 face_location 에서 얻은 값을 다음 변수에 집어넣는다.
                    face_image = small_frame[top:bottom, left:right]  # face_image는 다음과 같이 resize한 값이다.
                    face_image2 = rgb_small_frame[top:bottom, left:right]
                    # 인식된 얼굴 출력함.
                    face_image = cv2.resize(face_image, (150, 150))  # face_image는 Opencv를 이용하여 다음과 같이 resize 한다.
                    face_image2 = cv2.resize(face_image2, (150, 150))
                    pil_image = Image.fromarray(face_image2)
                    #pil_image.show()
                    # 새 얼굴을 등록한다.개
                    register_new_face(face_encoding, face_image)
                    pil_image.save(buffer, format='jpeg')
                    img_str = base64.b64encode(buffer.getvalue())
                    # 인식된 얼굴 전송
                    img_df = pd.DataFrame({'image_data': [img_str]})
                    img_df.to_sql('images', con=engine, if_exists='append', index=False)

                    # DB에 저장된 얼굴을 출력하도록 하는 부분
                    img_str = img_df['image_data'].values[0]
                    img = base64.decodebytes(img_str)
                    im = Image.open(BytesIO(img))
                    # data를 decode 하여 원래는 jpg로 만든다.
                    decimg = cv2.imdecode(data, 1)
                    # decode 한것을 my.jpg 파일로 만들기
                    cv2.imwrite(nowDate.strftime("/home/jae/PycharmProjects/UDP/registered_face/%Y-%m-%d %H:%M.jpg"),
                                decimg)
                    cv2.imwrite(nowDate.strftime("/home/jae/PycharmProjects/web_gallery/images/%Y-%m-%d %H:%M.jpg"),
                                decimg)

                face_labels.append(face_label)  # 반복문 이전에 만든 face_labels 리스트에 face_label(new visitor!)을 추가한다.

                # 얼굴 주위에 네모난 박스를 만드는 반복문이다.
                for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # 얼굴 주위에 박스를 만든다.
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)  # label color

                    # 얼굴 아래에 라벨을 만든다.
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, face_label, (left + 4, bottom - 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255
                                                                                                          ), 1)

                # 최근이 방문자의 방문횟수를 보여주는 반복문이다.
                number_of_recent_visitors = 0  # 현재 방문자를 카운트 하는 것.
                # known_face_metadata 리스트에 있는 모든 요소를 metadata로 잡아 반목문을 돌린다.
                for metadata in known_face_metadata:
                    # 5분이후 이 사람을 봤을 때 작동한다.
                    if datetime.now() - metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                        # known_face_image를 그린다.
                        x_position = number_of_recent_visitors * 150
                        frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                        number_of_recent_visitors += 1

                        # 얼마나 많이 방문했는지를 보여주는 코드이다.
                        visits = metadata['seen_count']
                        if visits == 1:
                            visit_label = "Registering!"
                            visits += 100
                        else:
                            visit_label = f"Registering Face"
                            # 얼굴 유사도를 보여주는 코드이다.
                            similarity = metadata['face_dist']
                            sim_per = round((1 - similarity), 3) * 100
                            sim_label = f" similarity:{sim_per} %"
                            #cv2.putText(frame, sim_label, (x_position, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                        #(255, 255, 255), 1)

                        cv2.putText(frame, visit_label, (x_position + 10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                    (0, 255, 0), 1)

                # 현재 방문자가 1명보다 많으면 다른 곳에 덧붙여 이를 출력한다.
                if number_of_recent_visitors > 1:
                    cv2.putText(frame, "Recognizing", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # 이미지를 보여주기 위해 사용하는 imshow 'Video는 윈도의 창의 제목, frame은 video_capture.read()한 값
                cv2.imshow('Recognizing', frame)

            conn.send('I'.encode('utf-8'))  # 이부분을 conn으로 변경하였다.

            if cv2.waitKey(1) & 0xFF == ord('q'):
                conn.send('I'.encode('utf-8'))  # 이부분을 conn으로 변경하였다.
                print('메세지 전송 완료')
                save_known_faces()
                break

            # crash 방지를 위해 save_known_face를 저장한다. 저장하기 위해 도는 if문
            if len(face_locations) > 0 and number_of_faces_since_save > 100:
                number_of_faces_since_save = 0
            else:
                number_of_faces_since_save += 1


if __name__ == "__main__":
    load_known_faces()
    #time.sleep(5)
    try:
        IP = '164.125.234.91'
        PORT1 = 5051
        PORT2 = 5060
        SIZE = 1024
        ADDR1 = (IP, PORT1)
        ADDR2 = (IP, PORT2)

        server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket1.bind(ADDR1)
        server_socket1.listen()

        server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket2.bind(ADDR2)
        server_socket2.listen()

        read_socket_list = [server_socket1, server_socket2]

        conn_read_socket_list, conn_write_socket_list, conn_except_socket_list = select.select(read_socket_list, [],
                                                                                                   [])
        for conn_read_socket in conn_read_socket_list:
            if conn_read_socket == server_socket1:
                client_socket, client_addr = server_socket1.accept()
                msg = client_socket.recv(SIZE)
                # 얼굴 등록하시겠습니까?
                if msg.decode('utf-8') == 'R':
                    print('hello socket1')
                    main_loop()

                client_socket.close()
            elif server_socket2 == server_socket2:
                print('hello socket2')
                client_socket, client_addr = server_socket2.accept()
                msg = client_socket.recv(SIZE)
                print("[{}] message : {}".format(client_addr, msg))
                client_socket.sendall("welcome 5060!".encode())
                client_socket.close()
    except ConnectionResetError:
        restart()
    except IndexError:
        print("얼굴 등록된 데이터가 없습니다. 등록 먼저 해주세요!")


