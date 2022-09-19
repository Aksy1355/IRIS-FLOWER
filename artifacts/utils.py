import numpy as np
import pickle

class ifp():
    def __init__(self,data):
        self.data=data

    def load_model(self):
        with open(r'artifacts/model.pkl','rb') as file:
            self.model=pickle.load(file)

    def predict(self):
        self.load_model()
        sepal_length = self.data["sepal_length"]
        sepal_width = self.data["sepal_width"]
        petal_length = self.data["petal_length"]
        petal_width = self.data["petal_width"]

        arr=np.array([sepal_length,sepal_width,petal_length,petal_width],ndmin=2,dtype='float')
        print(arr)

        res=self.model.predict(arr)
        print(res)
        return res
        
if __name__ == '__main__':
    data={
        "sepal_length":2.1,
        "sepal_width":5.2,
        "petal_length":2.3,
        "petal_width":1.6
    }

    ifp_obj=ifp(data)
    ifp_obj.predict()