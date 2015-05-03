import h5py

class RegressionModel:
    HDFS_FILEPATH = "../data/output.hdf5"

    def loadDataFromHDFS(self):
        fileObj = h5py.File(self.HDFS_FILEPATH, 'r')
        self.idList = fileObj["IDs"]
        self.features = fileObj["Features"]
        self.viralityList = fileObj["Virality"]
        print self.idList[60000]
        print len(self.idList)
        print self.features
        print self.viralityList
        print self.features[0]
        print self.features.attrs

def main():
    model = RegressionModel()
    model.loadDataFromHDFS()


if __name__ == "__main__":
    main()