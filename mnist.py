import lasagne.layers as ll
from lasagne.layers.conv import TransposedConv2DLayer, Conv2DLayer
import lasagne.nonlinearities as nonlin
import matplotlib.pyplot as plt
import lasagne.updates as lu
import lasagne.objectives as lo
import lasagne.regularization as lreg
import numpy as np
import theano
import theano.tensor as T
from loadmnist import load_dataset
from sklearn.utils import shuffle
gen = []
noise=T.matrix()
batch_size=512
input_shape=(batch_size,81)
decay_every=100
decay=0.99
X=load_dataset()
Xvar=T.tensor4("@!! image variable !!@")
noisevar = T.matrix("@!! generator input variable !!@")
def display_filters(layer):
    pass
def float32(n):
    return np.cast['float32'](n)

lr=theano.shared(float32(0.001/2))
minlr=0.0001
def show_shapes(layers):
    """Iterates through a l ist of lasagne
        shows the input and output shape for each layer
        won't work for nets that aren't feeforward probably"""
    shape=input_shape
    for l in layers:
        shape = l.get_output_shape_for(shape)
        #params = l.get_params()
        #shapes = [p.get_value().shape for p in params]
        #for i in shapes: n_params*=i
        print("layer {}, outputs shape {}".format(l,shape))

gen_filters=32
#define generator network
gen.append(ll.InputLayer(shape=input_shape))

gen.append(ll.batch_norm(ll.DenseLayer(gen[-1], num_units=1024,
	)))

gen.append(ll.batch_norm(ll.DenseLayer(gen[-1], num_units=gen_filters*7*7,
	)))

gen.append(ll.ReshapeLayer(gen[-1],shape=(batch_size,gen_filters,7,7)))



gen.append(ll.batch_norm(TransposedConv2DLayer(
    gen[-1],
    filter_size=(5,5),
    num_filters=gen_filters,
    stride=(2,2),
    crop=(2,2)
   )))


gen.append(ll.batch_norm(TransposedConv2DLayer(
    gen[-1],
    filter_size=(4,4),
    num_filters=1,
    stride=(2,2),
    crop=(0,0),nonlinearity=nonlin.sigmoid
   )))




gen.append(ll.ReshapeLayer(gen[-1],shape=(batch_size,1,28,28)))
gen_out=ll.get_output(gen[-1],noisevar,deterministic=True)
gen_params=ll.get_all_params(gen)

get_image=theano.function(inputs=[noisevar],outputs=gen_out,
        allow_input_downcast=True)
#show_shapes(gen)
#define discriminator network
disc=[]
disc.append(ll.InputLayer(shape=(None,1,28,28)))
disc.append(ll.dropout(Conv2DLayer(disc[-1],num_filters=gen_filters,stride=(2,2),filter_size=(5,5),
	nonlinearity=nonlin.very_leaky_rectify),p=0.25))

disc.append(ll.dropout(Conv2DLayer(disc[-1],num_filters=64,
	filter_size=(5,5),
	nonlinearity=nonlin.very_leaky_rectify),p=0.35))


#disc.append(ll.dropout(ll.DenseLayer(disc[-1],num_units=1000),p=0.3))
#disc.append(ll.dropout(ll.DenseLayer(disc[-1],num_units=500),p=0.2))
#disc.append(ll.dropout(ll.DenseLayer(disc[-1],num_units=250),p=0.3))
#disc.append(ll.GaussianNoiseLayer(disc[-1], sigma=0.01))
disc.append(ll.dropout(ll.DenseLayer(disc[-1], num_units=512,nonlinearity=nonlin.very_leaky_rectify),p=0.0))


disc.append(ll.DenseLayer(disc[-1],num_units=1,nonlinearity=nonlin.sigmoid))
disc_data=ll.get_output(disc[-1],inputs=Xvar)
disc_gen=ll.get_output(disc[-1],gen_out)
disc_params=ll.get_all_params(disc)

#data_obj=T.mean(T.log(disc_data)) #objective function for data
data_obj=lo.binary_crossentropy(disc_data,T.ones(batch_size)).mean()



data_train=theano.function(
    inputs=[Xvar],
    outputs=data_obj,
    updates=lu.adam(data_obj,disc_params,learning_rate=lr),
        allow_input_downcast=True
    )

#gen_obj = T.mean(T.log(T.ones(batch_size) - disc_gen )  )
gen_obj=lo.binary_crossentropy(disc_gen,T.ones(batch_size)).mean()
b=theano.function(inputs=[noisevar],outputs=disc_gen,
        allow_input_downcast=True)

u_obj=lo.binary_crossentropy(disc_gen,T.zeros(batch_size)).mean()
disc_train = theano.function(
        inputs=[noisevar],
        outputs=u_obj,
        updates=lu.adam(u_obj,disc_params,learning_rate=lr),
        allow_input_downcast=True
)

gen_train = theano.function(
        inputs=[noisevar],
        outputs=gen_obj,
        updates=lu.adam(gen_obj,gen_params,learning_rate=lr/15),
        allow_input_downcast=True
)


def plot_gen_outputs(noise,n):
    images=get_image(noise)
    for i in range(n):
        index=np.random.randint(low=0,high=128)
        print("plotting idx: {}, input: {}".format(index,noise[index].max()))
        plt.imshow(images[index].reshape(28,28),cmap='gray')
        print("plotted")

        plt.pause(0.0001)


#train loop
n_iter=100000;v=0;n=1

print("Starting training")


while n < n_iter:
    if v > X.shape[0] / batch_size:
        v=0;X=shuffle(X,random_state=47)

    batch=X[v:(v+batch_size)]; v = v + batch_size
    noise=np.random.normal(size=input_shape)
    CurDataObj=data_train(batch)
    CurGenObj=disc_train(noise)
    if np.isnan(CurDataObj):
        pass
        #execfile('mnist.py')
    _ = gen_train(noise)
    preds=b(noise)
    print("i: {} | disc objective on data: {} |  gen percent: {}".format(n,CurDataObj,preds.mean()))
    if n % 40 == 0:
        plot_gen_outputs(noise,1)
    n+=1
    if n % decay_every == 0 and lr.get_value() < minlr:

        lr.set_value(float32(lr.get_value()*decay))
        print("decayed lr to {}".format(lr.get_value()))
