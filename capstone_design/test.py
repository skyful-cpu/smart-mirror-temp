from bluetooth import *

mac = [
    '98:D3:31:FB:86:EE',
    '98:DA:60:01:C3:37'
    ]

iot = ['light', 'boiler', 'fan']
max_iot_len = 3

socket = BluetoothSocket( RFCOMM )
socket.connect((mac[0], 1))
print("bluetooth connected")
socket.send('i')
print('message sent to arduino')

count = 0
idx = 0
return_dict = {}
print('receiving data...')

while True:
    byte_data = socket.recv(2048)
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

print('received from arduino')
print(return_dict)
socket.close()
print('socket closed')