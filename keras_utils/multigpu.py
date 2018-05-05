from keras import backend as K
from keras import Model
from keras.utils import multi_gpu_model


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def get_number_of_gpus():
    devices = [x.name for x in K.get_session().list_devices()]
    # ['/cpu:0', '/gpu:0', '/gpu:1']
    return len([1 for d in devices if 'gpu' in d.lower()])

# # model topology instantiation above
# ser_model = Keras.models.Model(inputs = x, output=out)
# parallel_model = ModelMGPU(ser_model , 4)
#
# #callback to save best weights
# mod_wt_path = './best_weights.hdf5'
# best_wts_callback = callbacks.ModelCheckpoint(mod_wt_path, save_weights_only=True, save_best_only=True)
#
# # compile the parallel model prior to fit
# parallel_model.fit(X, y, callbacks=[best_wts_callback])
#
# #Now I want to infer on single GPU so I load saved weights ??
# ser_model.load_weights(mod_wt_path)
#
# # I think you might have to compile the serial model prior to predict
# ser_model.predict(X_holdout)
