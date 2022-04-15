class DataConstants:
    def __init__(self, eTag):
        if eTag in ['e80', 'e83', 'e84', 'e99']:
            self.indicator = 'PM2.5'
            self.cities = ["xian", 'chengdu']
        if eTag in ['e82', 'e85', 'e86']:
            self.indicator = 'PM10'
            self.cities = ["xian", 'chengdu']
        if eTag in ['e91', 'e93', 'e95', 'e98']:
            self.indicator = 'PM2.5'
            self.cities = ["xian"]

        if eTag == 'e84':
            # this is especially used for 'e84', '1470A' is removed for Xi'an，'1438A' is removed for Chengdu
            self.stations = [
                ['1462A', '1463A', '1464A', '1465A', '1466A', '1467A', '1468A', '1469A', '1471A', '1472A', '1473A',
                 '1474A'], ['1431A', '1432A', '1433A', '1434A', '1437A', '2880A', '3136A']]
        else:
            self.stations = [
                ['1462A', '1463A', '1464A', '1465A', '1466A', '1467A', '1468A', '1469A', '1470A', '1471A', '1472A',
                 '1473A', '1474A'], ['1431A', '1432A', '1433A', '1434A', '1437A', '1438A', '2880A', '3136A']]

    def getPublicData(self):
        return self.cities, self.indicator, self.stations
