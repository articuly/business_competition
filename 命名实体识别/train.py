import bilsm_crf_model
import process_data
import numpy as np

EPOCHS = 3
model, (train_x, train_y) = bilsm_crf_model.create_model()
model.fit(train_x, train_y,batch_size=640,epochs=EPOCHS)
model.save('model/crfa3.h5')
