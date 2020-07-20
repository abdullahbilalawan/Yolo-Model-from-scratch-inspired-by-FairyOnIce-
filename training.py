from yolo_data import *
from model_file import *




### The location where the VOC2012 data is saved.
train_image_folder = r"C:\Users\DELL\PycharmProjects\Yolo\JPEGImages\\"
train_annot_folder = r"C:\Users\DELL\PycharmProjects\Yolo\Annotations\\"

np.random.seed(1)
from backend import parse_annotation
train_image, seen_train_labels = parse_annotation(train_annot_folder,
                                                  train_image_folder,
                                                  labels=LABELS)


from backend import SimpleBatchGenerator

BATCH_SIZE        = 200
IMAGE_H, IMAGE_W  = 416, 416
GRID_H,  GRID_W   = 13 , 13
TRUE_BOX_BUFFER   = 50
BOX               = int(len(ANCHORS)/2)

generator_config = {
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}


def normalize(image):
    return image / 255.
train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                             norm=normalize, shuffle=True)



from backend import define_YOLOv2, set_pretrained_weight, initialize_weight
CLASS             = len(LABELS)
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS,
                                  trainable=False)

GRID_W = 13
GRID_H = 13
BATCH_SIZE = 34
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0


def custom_loss(y_true, y_pred):
    return (custom_loss_core(
        y_true,
        y_pred,
        true_boxes,
        GRID_W,
        GRID_H,
        BATCH_SIZE,
        ANCHORS,
        LAMBDA_COORD,
        LAMBDA_CLASS,
        LAMBDA_NO_OBJECT,
        LAMBDA_OBJECT))
BATCH_SIZE   = 32
generator_config['BATCH_SIZE'] = BATCH_SIZE

early_stop = EarlyStopping(monitor='loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

model.fit_generator(generator        = train_batch_generator,
                    steps_per_epoch  = len(train_batch_generator),
                    epochs           = 50,
                    verbose          = 1,
                    #validation_data  = valid_batch,
                    #validation_steps = len(valid_batch),
                    callbacks        = [early_stop],
                    max_queue_size   = 3)