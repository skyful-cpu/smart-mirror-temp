from bluetooth import *
import time

commands = [
    '불켜줘',
    '불꺼줘',
    '조명켜줘',
    '조명꺼줘',
    '가스잠가줘',
    '가스밸브잠가줘',
    '보일러켜줘',
    '보일러꺼줘',
    '창문열어줘',
    '창문닫아줘',
    '커튼쳐줘',
    '커튼걷어줘',
    '환풍기켜줘',
    '환풍기꺼줘',
    '스마트미러명령어'
]

# 각 IoT 기기에 사용된 블루투스 모듈의 MAC 주소
mac = [
    '98:D3:31:FB:86:EE',
    '98:DA:60:01:C3:37'
    ]

# IoT 기기 종류
iot = ['light', 'boiler', 'fan', 'window', 'valve']
max_iot_len = len(iot)

class IoT:
    
    # command를 인자로 받아와 클래스 생성
    def __init__(self, command=None):
        self.socket_busy = False
        # self.socket = BluetoothSocket( RFCOMM )
        #self.current_iot_state = self.get_initial_state()

        if command:
            self.command = command
        else:
            self.command = ''
        
        
        
    # 최초 접속 시 IoT on / off 여부 확인
    def get_initial_state(self):
        '''
        웹 페이지 최초 접속 시 블루투스 통신으로 IoT on / off 여부 확인

        결과를 dictionary {'IoT1': 'state', ...} 로 반환
        '''
        return_dict = {}

        # if self.socket_busy:
        #     self.socket.close()
        #     print('socket closed because socket is busy')
        #     self.socket_busy = False
        #     return return_dict

        self.socket_busy = True

        self.socket = BluetoothSocket( RFCOMM )
        self.socket.connect((mac[0], 1))
        self.socket.send('i')

        idx = 0

        while True:
            byte_data = self.socket.recv(2048)
            data = byte_data.decode('utf-8')

            if data is ' ':
                continue
            elif data is 'q':
                break
            else:
                if idx >= max_iot_len:
                    return_dict = {}
                    break
                return_dict[iot[idx]] = data
                idx += 1

        # while True:
        #     byte_data = self.socket.recv(2048)
        #     data = byte_data.decode('utf-8')

        #     if count == 0:
        #         if data is not 'q':
        #             continue
        #         else:
        #             count += 1
        #             continue
        #     elif count == 1:
        #         if data is ' ':
        #             continue
        #         elif data is 'q':
        #             break
        #         else:
        #             #print(data)
        #             if idx >= max_iot_len:
        #                 return_dict = {}
        #                 break
        #             return_dict[iot[idx]] = data
        #             idx += 1
            
        self.socket.close()
        print('update successed socket closed')
        self.socket_busy = False
        print(return_dict)
        return return_dict
    
    # command에 따른 IoT 제어
    def control_iot(self):
        '''
        블루투스로 IoT 기기를 제어하는 함수
        어느 IoT를 제어했는 지에 대한 정보를 dictionary 형태로 반환
        '''
        
        if (self.command == commands[0]) or (self.command == commands[1]) or (self.command == commands[2]) or (self.command == commands[3]):
            result = self.control_light(self.command)
            return result
            
        elif (self.command == commands[4]) or (self.command == commands[5]):
            result = self.control_valve(self.command)
            return result
            
        elif (self.command == commands[6]) or (self.command == commands[7]):
            result = self.control_boiler(self.command)
            return result
            
        elif (self.command == commands[8]) or (self.command == commands[9]):
            result = self.control_window(self.command)
            return result

        elif (self.command == commands[10]) or (self.command == commands[11]):
            result = self.control_curtain(self.command)
            return result
            
        elif (self.command == commands[12]) or (self.command == commands[13]):
            result = self.control_fan(self.command)
            return result
        
    # control light    
    def control_light(self, cmd):
        '''
        return : control results in dictionary
        
        cmd : smart mirror command
        '''
        print(f'command : {cmd}')
        
        if cmd == (commands[0] or commands[2]):
            # send 1 to turn on the light
            self.send_bluetooth(mac[0], '1')
            return {'light' : 'on'}
            
        else:
            # send 0 to turn off the light
            self.send_bluetooth(mac[0], '0')
            return {'light' : 'off'}

    # control gas valve   
    def control_valve(self, cmd):
        # send 10 to close the valve
        print(f'command : {cmd}')
        self.send_bluetooth(mac[0], 'v')
        return {'valve' : 'off'}

    # control boiler // 블루투스 모듈이 부족해서 지금은 LED에 쓰는 모듈이랑 같이 사용  
    def control_boiler(self, cmd):
        print(f'command : {cmd}')
        
        if cmd == commands[6]:
            # send 3 to turn on the boiler
            self.send_bluetooth(mac[0], '3')
            return {'boiler' : 'on'}
        else:
            # send 2 to turn off the boiler
            self.send_bluetooth(mac[0], '2')
            return {'boiler' : 'off'}

    # control window   
    def control_window(self, cmd):
        print(f'command : {cmd}')
        
        if cmd == commands[8]:
            # send 3 to open the window
            self.send_bluetooth(mac[0], '7')
            return {'window' : 'on'}
        else:
            # send 2 to close the window
            self.send_bluetooth(mac[0], '6')
            return {'window' : 'off'}

    # control curtain   
    def control_curtain(self, cmd):
        print(f'command : {cmd}')
        
        if cmd == commands[10]:
            # send 3 to on the curtain
            self.send_bluetooth(mac[0], '9')
            return {'curtain' : 'on'}
        else:
            # send 2 to off the curtain
            self.send_bluetooth(mac[0], '8')
            return {'curtain' : 'off'}

    # control fan   
    def control_fan(self, cmd):
        print(f'command : {cmd}')
        
        if cmd == commands[12]:
            # send 3 to turn on the boiler
            self.send_bluetooth(mac[0], '5')
            return {'fan' : 'on'}
        else:
            # send 2 to turn off the boiler
            self.send_bluetooth(mac[0], '4')
            return {'fan' : 'off'}
    
    # connect IoT via bluetooth
    def send_bluetooth(self, mac, msg):
        
        # 블루투스 통신을 위한 소켓 객체을 만들고 지정한 MAC 주소로 연결
        time.sleep(0.5)
        self.socket.close()
        self.socket = BluetoothSocket( RFCOMM )
        self.socket.connect((mac, 1))
        
        # 음성 인식 결과에 따라 다른 메시지를 아두이노로 전송
        self.socket.send(msg)

        self.socket.close()
        self.socket_busy = False
        print('socket closed')