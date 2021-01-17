import os
import numpy as np
from spektral.data import Graph
from spektral.data.dataset import Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from spektral.layers import GCNConv, GlobalSumPool,ECCConv
from spektral.data import BatchLoader,DisjointLoader
from tensorflow.keras import losses,optimizers,metrics
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate = 1e-4

seed=1

def load_data():
    featurearr=np.load('/home/som/lab/seed-yzj/newpaper4/laboratory/featurenpy/feature.npy')
    simarr=np.load('/home/som/lab/seed-yzj/newpaper4/laboratory/featurenpy/simlarity.npy')
    filearr=np.load('/home/som/lab/seed-yzj/newpaper4/laboratory/datanpy/hr-gt-each10s-laboratory.npy')
    labelarr=[]
    reshapefeaturearr=np.zeros((featurearr.shape[0],featurearr.shape[1],featurearr.shape[2]*featurearr.shape[3]*featurearr.shape[4]))

    for j in range(filearr.shape[0]):
        labelarr.append(float(filearr[j][3]))
        for i in range(featurearr.shape[1]):
            # print(np.array(featurearr[j,i]).flatten().shape)
            reshapefeaturearr[j,i]=np.array(featurearr[j,i]).flatten()
    labelarr = np.array(labelarr)
    print(reshapefeaturearr.shape,simarr.shape,labelarr.shape)
    return reshapefeaturearr,simarr,labelarr

class MyDataset(Dataset):

    def __init__(self, xarr,yarr,aarr,edge_attrarr, **kwargs):

        self.xarr = xarr
        self.yarr = yarr
        self.aarr = aarr
        self.edge_attrarr = edge_attrarr
        self.num=xarr.shape[0]

        super().__init__(**kwargs)

    def read(self):
        def make_graph(n):
            x = self.xarr[n]
            y = self.yarr[n]
            a = self.aarr[n]
            e = self.edge_attrarr[n]
            g=Graph(x=x, y=y, a=a,e=e)
            return g

        # We must return a list of Graph objects
        return [make_graph(_) for _ in range(self.num)]

def graphdatageneration(featurearr, simarr, labelarr):
    xarr = []
    yarr = []
    aarr = []
    edge_attrarr = []
    for i in range(featurearr.shape[0]):
        x = featurearr[i]
        a = np.ones((8, 8))
        y = labelarr[i]
        edge_att = np.array(simarr[i]).flatten()
        edge_att = [[j] for j in edge_att]
        # x,y,a,edge_attr=graphgeneration(featurearr[i], simarr[i], labelarr[i])
        xarr.append(x)
        yarr.append([y])
        aarr.append(a)
        edge_attrarr.append(edge_att)

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    aarr = np.array(aarr)
    edge_attrarr = np.array(edge_attrarr)
    return xarr, yarr, aarr, edge_attrarr

def buildmodel(dataset):

    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    n_out = dataset.n_labels  # Dimension of the target

    #model
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None,), sparse=True, name="A_in")
    E_in = Input(shape=(S,), name="E_in")
    I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

    X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
    X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
    X_3 = GlobalSumPool()([X_2, I_in])
    output = Dense(n_out)(X_3)

    # Build model
    model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
    model.summary()

    return model


def pipeline():
    featurearr, simarr, labelarr=load_data()
    xarr, yarr, aarr, edge_attrarr=graphdatageneration(featurearr, simarr, labelarr)

    dataset = MyDataset(xarr,yarr,aarr,edge_attrarr)

    np.random.seed(10)
    # Train/test split
    idxs = np.random.permutation(len(dataset))
    split = int(0.8 * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]
    loader_tr = DisjointLoader(dataset_tr, batch_size=32, epochs=30,shuffle=True)
    loader_te = DisjointLoader(dataset_te, batch_size=32, epochs=1,shuffle=True)

    model=buildmodel(dataset)

    opt = optimizers.Adam(lr=learning_rate)
    loss_fn = losses.MeanSquaredError()


    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
            mae=losses.MeanAbsoluteError()(target, predictions)
            mape=losses.MeanAbsolutePercentageError()(target, predictions)

            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss,mae,mape

    print("training")
    current_batch = 0
    model_loss = 0
    total_mape=0
    total_mae=0
    for batch in loader_tr:
        outs,mae,mape= train_step(*batch)

        model_loss += outs
        total_mae+=mae
        total_mape+=mape
        current_batch += 1
        if current_batch == loader_tr.steps_per_epoch:
            print("MSE: {}".format(model_loss / loader_tr.steps_per_epoch),
                  "MAE: {}".format(total_mae/ loader_tr.steps_per_epoch),
                  "MAPE: {}".format(total_mape/ loader_tr.steps_per_epoch))
            model_loss = 0
            total_mae = 0
            total_mape = 0
            current_batch = 0


    print("testing")
    model_loss = 0
    model_mae=0
    model_mape = 0
    for batch in loader_te:
        inputs, target = batch
        predictions = model(inputs, training=False)
        model_loss += loss_fn(target, predictions)
        model_mae += losses.MeanAbsoluteError()(target, predictions)
        model_mape+= losses.MeanAbsolutePercentageError()(target, predictions)

    model_loss /= loader_te.steps_per_epoch
    model_mae /= loader_te.steps_per_epoch
    model_mape /= loader_te.steps_per_epoch
    print("Done. Test MSE: {}".format(model_loss),
          "Test MAE: {}".format(model_mae),
          "Test MAPE: {}".format(model_mape))
    model.save('/home/som/lab/seed-yzj/newpaper4/laboratory/model/fusion.hdf5')


if __name__ == '__main__':
    pipeline()