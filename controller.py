
from msilib.schema import SelfReg
import snap7
from snap7.common import check_error, load_library, ipv4
from snap7.exceptions import Snap7Exception
import numpy as np
import os
import sys
import time

class plcS7:
    def __init__(self, ip, rack, slot):
        
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.bit2plc = 0
        self.HoB = 0
        self.stateCam = 0
        self.plc = snap7.client.Client()

    def sendData(self, distanceArr):
        self.plc.connect( self.ip, self.rack, self.slot )
        

        # En el PLC
        '''dbDistance = 1
        startDistance = 12
        offsDistance  = 0 

        dbStateC   = 2
        startStatec = 8
        offsStatec  = 0 

        dbHoc      = 2
        startHoc   = 11
        offsHoc  = 0  '''

        dbDistance = 1
        startDistance = 0
        offsDistance  = 0 

        dbStateC   = 1
        startStatec = 2
        offsStatec  = 0 

        dbHoc      = 1
        startHoc   = 2
        offsHoc  = 1

        data = bytearray(2)
        dataError = bytearray(1)
        #print(data)

        sumArr = np.array(distanceArr)

        dbRead = self.plc.db_read( dbDistance, startDistance, 1)
        snap7.util.set_int(data, offsDistance, int(sumArr.sum()))
        self.plc.db_write(dbDistance, startDistance, data)

        dbRead = self.plc.db_read( dbStateC, startStatec, 1)
        snap7.util.set_bool(dataError, 0, offsStatec, self.stateCam)
        self.plc.db_write(dbStateC, startStatec, dataError)        

        dbRead = self.plc.db_read( dbHoc, startHoc, 1)
        snap7.util.set_bool(dataError, 0, offsHoc, self.HoB)
        self.plc.db_write(dbHoc, startHoc, dataError)

        return 0