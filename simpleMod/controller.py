import snap7
from snap7.common import check_error, load_library, ipv4
from snap7.exceptions import Snap7Exception

class plcS7:
    def __init__(self, ip, rack, slot):
        
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.bit2plc = 0
        self.HoB = 0
        self.stateCam = 0
        self.plc = snap7.client.Client()
    
    def sendData(self, dist):
        
        self.plc = snap7.client.Client()
        self.plc.connect( self.ip, self.rack, self.slot )

        data = bytearray(2)
        dataError = bytearray(1)

        snap7.util.set_int(data,0, dist)
        self.plc.db_write(1, 12, data)

        snap7.util.set_bool(dataError, 0, 0, self.stateCam)
        self.plc.db_write(2, 8, dataError)        

        snap7.util.set_bool(dataError, 0, 0, self.HoB)
        self.plc.db_write(2, 11, dataError)

        return 0