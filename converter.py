class Converter:
    def __init__(self):
        self.categoryIndex={}
        self.oneItem={}
        self.ID=0
        self.display=""

    def convert(self, filename):
        f=open(filename, 'r')
        for x in f.readlines():
            #print(x)
            if 'id' in x:
                TEST= x.split(":")
                TEST[1]=TEST[1].rstrip('\n')
                self.ID= TEST[1]
            if 'display_name' in x:
                TEST= x.split(":")
                TEST[1] = TEST[1].rstrip('\n')
                self.display= TEST[1]
            if "}" in x:
                self.categoryIndex[int(self.ID)]= {'id':  int(self.ID), 'name': self.display}
                self.ID=0
                self.display=""
            #print(self.categoryIndex)

    def getCategoryIndex(self):
        return self.categoryIndex

#
#converter=Converter()
#converter.convert('mscoco_label_map.pbtxt')
