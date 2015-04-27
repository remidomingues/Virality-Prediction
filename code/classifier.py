import h5py

class Classifier:

    def loadDataFromHDFS(self, filename):
        fileObj = h5py.File(filename, 'r')
        self.idList = fileObj["IDs"] 
        self.features = fileObj["Features"]
        self.viralityList = fileObj["Virality"]
        # print self.idList[60000]
        # print len(self.idList)
        # print self.features
        # print self.viralityList
        # print self.features[0]
        # print self.features.attrs

def main():
    cl = Classifier()
    hdfsFilename = "../data/output.hdf5"
    cl.loadDataFromHDFS(hdfsFilename)


if __name__ == "__main__":
    main()