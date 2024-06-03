import serial

#This communicates with the serial output of the temperature probes used

def find_probes(): 
    #Ateempt to connect to temperature probes
    device = '/dev/ttyUSB0' #Permissions may be denied for this, you may have to alter them on your device
    canconnect = True 
    try:
        serial.Serial(device, 9600, bytesize = serial.SEVENBITS, parity = serial.PARITY_EVEN, stopbits = serial.STOPBITS_ONE, dsrdtr = False, timeout = 0.1)
    except:
        canconnect = False
        print('Unable to connect to Temperature probes - check permissions for /dev/ttyUSB0 if the device is plugged in')
    if canconnect:
        ser = serial.Serial(device, 9600, bytesize = serial.SEVENBITS, parity = serial.PARITY_EVEN, stopbits = serial.STOPBITS_ONE, dsrdtr = False, timeout = 0.1)
        return canconnect, ser
    else:
        return canconnect, None
    #We use canconnect to test if the temperature probes are available - if they are we read ser, the serial input

def get_temps(ser):
    #Reads the serial output string to retrieve the temperature read by probe 1 (t1) and probe 2 (t2)
    ser.write(b"CRDG? 0\r\n")
    tempss = ((ser.readline()).decode('utf-8')[:-2]).split(',')
    T1 = float(tempss[4][1:])
    T2 = float(tempss[5][1:])
    return T1, T2
    