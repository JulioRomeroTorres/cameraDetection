
from msilib.schema import SelfReg
import snap7
from snap7.common import check_error, load_library, ipv4
from snap7.exceptions import Snap7Exception
import os
import sys
import time

class plcS7:
    def __init__(self, rack, slot, url):
        self.rack = rack
        self.slot = slot
        self.bit2plc = 0
        self.plc = snap7.client.Client()

    def sendData():

    
        plc.connect(IP, RACK, SLOT)
        data = bytearray(2)
        print(data)
        data_error = bytearray(1)

        #Envio de la distancia y estado de la Camara 2
        snap7.util.set_int(data,0, lon)#v
        plc.db_write(1, 12, data)

        snap7.util.set_bool(data_error, 0, 0, e2)
        plc.db_write(2, 8, data_error)        

        ##Heart bit, para saber si el programa esta vivo
        snap7.util.set_bool(data_error, 0, 0, Heart_Bit_to_PLC)
        plc.db_write(2, 11, data_error)