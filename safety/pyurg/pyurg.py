#!/usr/bin/env python
# -*- coding:utf-8 -*-

# The MIT License
#
# Copyright (c) 2010 Yota Ichino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import serial
import re
import math
import time

class UrgDevice(serial.Serial):
    def __init__(self):
        super(serial.Serial, self).__init__()

    def __del__(self):
        self.laser_off()

    def connect(self, port = '/dev/ttyACM0', baudrate = 9600, timeout = 0.1):
        '''
        Connect to URG device
        port      : Port or device name. ex:/dev/ttyACM0, COM1, etc...
        baudrate  : Set baudrate. ex: 9600, 38400, etc...
        timeout   : Set timeout[sec]
        '''
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        try:
            self.open()
        except:
            return False
        
        if not self.isOpen():
            return False

        self.set_scip2()
        for i in range(100):
            if self.get_parameter():
                break
        return True

    def flush_input_buf(self):
        '''Clear input buffer.'''
        self.flushInput()

    def send_command(self, cmd):
        '''Send command to device.'''
        self.write(cmd.encode())

    def __receive_data(self):
        read_data = self.readlines()
        read_data = [x.decode("utf-8") for x in read_data]
        return read_data
    
    def set_scip2(self):
        '''Set SCIP2.0 protcol'''
        self.flush_input_buf()
        self.send_command('SCIP2.0\n')
        return self.__receive_data()

        self.flush_input_buf()
        self.send_command('VV\n')
        get = self.__receive_data()
        return get

    def get_parameter(self):
        '''Get device parameter'''
        if not self.isOpen():
            return False
        
        self.send_command('PP\n')
        
        get = self.__receive_data()
        
        # check expected value
        if not (get[:2] == ['PP\n', '00P\n']):
            print("Parameter error in get_parameter", get)
            return False
        
        # pick received data out of parameters
        self.pp_params = {}
        for item in get[2:10]:
            tmp = re.split(':|;', item)[:2]
            self.pp_params[tmp[0]] = tmp[1]
        return self.pp_params

    def laser_on(self):
        '''Turn on the laser.'''
        if not self.isOpen():
            return False
        
        self.send_command('BM\n')
        
        get = self.__receive_data()

        if not(get == ['BM\n', '00P\n', '\n']) and not(get == ['BM\n', '02R\n', '\n']):
            return False
        return True
        
    def laser_off(self):
        '''Turn off the laser.'''
        open = self.isOpen()
        if not open:
            return False

        self.flush_input_buf()
        self.send_command('QT\n')
        get = self.__receive_data()
        
        if not(get == ['QT\n', '00P\n', '\n']):
            return False
        return True
    
    def __decode(self, encode_str):
        '''Return a numeric which converted encoded string from numeric'''
        decode = 0
        
        for c in encode_str:
            decode <<= 6
            decode &= ~0x3f
            decode |= ord(c) - 0x30
            
        return decode

    def __decode_length(self, encode_str, byte):
        '''Return leght data as list'''
        data = []
        
        for i in range(0, len(encode_str), byte):
            split_str = encode_str[i : i+byte]
            data.append(self.__decode(split_str))
            
        return data

    def index2rad(self, index):
        '''Convert index to radian and reurun.'''
        rad = (2.0 * math.pi) * (index - int(self.pp_params['AFRT'])) / int(self.pp_params['ARES'])
        return rad

    def create_capture_command(self):
        '''create capture command.'''
        cmd = 'GD' + self.pp_params['AMIN'].zfill(4) + self.pp_params['AMAX'].zfill(4) + '01\n'
        return cmd

    def scan_sec(self):
        '''Return time of a cycle.'''
        rpm = float(self.pp_params['SCAN'])
        return (60.0 / rpm)
        
    def capture(self):
        if not self.laser_on():
            print("Laser not on returning -1")
            return [], -1

        # Receive lenght data
        cmd = self.create_capture_command()
        self.flush_input_buf()
        self.send_command(cmd)
        time.sleep(0.1)
        get = self.__receive_data()
        
        # checking the answer
        if not (get[:2] == [cmd, '00P\n']):
            print("got weird answer from laser scanner returning -1")
            return [], -1
        
        # decode the timestamp
        tm_str = get[2][:-1] # timestamp
        timestamp = self.__decode(tm_str)
        
        # decode length data
        length_byte = 0
        line_decode_str = ''
        if cmd[:2] == ('GS' or 'MS'):
            length_byte = 2
        elif cmd[:2] == ('GD' or 'MD'):
            length_byte = 3
        # Combine different lines which mean length data
        NUM_OF_CHECKSUM = -2
        for line in get[3:]:
            line_decode_str += line[:NUM_OF_CHECKSUM]

        # Set dummy data by begin index.
        self.length_data = [-1 for i in range(int(self.pp_params['AMIN']))]
        self.length_data += self.__decode_length(line_decode_str, length_byte)
        return (self.length_data, timestamp)
        

def main():
    urg = UrgDevice()
    if not urg.connect():
        print('Connect error')
        exit()

    for i in range(10):
        data, tm = urg.capture()
        if data == 0:
            continue
        print(len(data), tm)



if __name__ == '__main__':
    main()
