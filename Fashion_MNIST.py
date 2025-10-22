####################################################################
# Function : load_data
# Inputs: val_split (float)- validation split ratio
# Outputs: (train_data,val_data,test_data)tuples
# Description: Loads Fashion-MNIST dataset,normalizes and splits into train,val,test sets.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

SEED = 42
ARTIFACT_DIR = "artifacts"
BEST_MODEL = "artifacts/best_model.keras"
FINAL_MODEL = "artifacts/final_model.keras"
FASHION_CLASSES = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

####################################################################
# Function : load_data
# Inputs: val_split (float)- validation split ratio
# Outputs: (train_data,val_data,test_data)tuples
# Description: Loads Fashion-MNIST dataset,normalizes and splits into train,val,test sets.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def load_data(val_split=0.1)-> tuple[Tuple[np.ndarray,np.ndarray],
                                     Tuple[np.ndarray,np.ndarray],
                                     Tuple[np.ndarray,np.ndarray]]:
    (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    x_train,x_val,y_train,y_val=train_test_split(
        x_train,y_train,test_size=val_split,random_state=SEED,stratify=y_train
    )
    x_train=(x_train.astype("float32")/255.0)[...,None]
    x_val=(x_val.astype("float32")/255.0)[...,None]
    x_test=(x_test.astype("float32")/255.0)[...,None]
    return (x_train,y_train),(x_val,y_val),(x_test,y_test)

####################################################################
# Function : load_flattened
# Inputs: None
# Outputs: Flattened train and test arrays with labels
# Description: Loads Fashion_MNIST and flattens images for classical ML models.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def load_flattened():
    (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    x_train=x_train.reshape(len(x_train),-1).astype("float32")/255.0
    x_test=x_test.reshape(len(x_test),-1).astype("float32")/255.0
    return x_train,y_train,x_test,y_test

####################################################################
# Function : build_cnn
# Inputs: lr(float)-learning rate
# Outputs: Compiled Keras CNN model
# Description: Builds and compiles CNN with augmentation,Conv2D, Dropout,BatchNorm,Dense.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def build_cnn(lr=1e-3)->keras.Model:
    data_augmentation=keras.Sequential(
        [
            layers.RandomTranslation(0.05,0.05,fill_mode="nearest"),
            layers.RandomRotation(0.05,fill_mode="nearest"),
            layers.RandomZoom(0.05,0.05,fill_mode="nearest"),
        ],
        name="augmentation",
    )
    inputs=keras.Input(shape=(28,28,1))
    x=data_augmentation(inputs)
    x=layers.Conv2D(32,3,padding="same",activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(32,3,padding="same",activation="relu")(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Dropout(0.25)(x)
    
    x=layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Dropout(0.25)(x)
    
    x=layers.Flatten()(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.40)(x)
    outputs=layers.Dense(10,activation="softmax")(x)
    
    model=keras.Model(inputs,outputs,name="fashion_cnn")
    opt=keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

####################################################################
# Function : train_and_evaluate
# Inputs: batch_size(int),epochs(int),lr(float)
# Outputs: Trained model + saved artifacts
# Description: Train CNN on Fashion-MNIST,evaluates saves curves,confusion matrix,misclassifications,report
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def train_and_evaluate(batch_size=128,epochs=15,lr=1e-3):
    ensure_dir(ARTIFACT_DIR)
    (x_train,y_train),(x_val,y_val),(x_test,y_test)=load_data(val_split=0.1)
    model=build_cnn(lr=lr)
    model.summary()
    callbacks=[
        keras.callbacks.ModelCheckpoint(BEST_MODEL,monitor="val_accuracy",save_best_only=True,
                                        verbose=1),
        keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True,monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-5,verbose=1),
    ]
    history=model.fit(
        x_train,y_train,
        validation_data=(x_val,y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
    print(f"[TEST] loss:{test_loss:.4f} | acc:{test_acc:.4f}")
    
    model.save(FINAL_MODEL)
    # plot_training_curves(history,ARTIFACT_DIR)
    # y_pred=model.predict(x_test,batch_size=256).argmax(axis=1)
    # plot_confusion_matrix_report(y_test,y_pred,ARTIFACT_DIR)
    # show_misclassifications(x_test,y_test,y_pred,limit=25,out_dir=ARTIFACT_DIR)
    # save_label_map(ARTIFACT_DIR)
    # save_summary(test_acc,test_loss,len(history.history["loss"]),ARTIFACT_DIR)
    
####################################################################
# Function : inference_grid
# Inputs: n_samples(int),seed(int)
# Outputs: Saved inference grid image
# Description: Loads saved model,predicts on random test samples,saves predictions grid.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def inference_grid(n_samples=9,seed=7):
    if not os.path.exists(BEST_MODEL):
        print(f"Could not find {BEST_MODEL}.Train first with --train.")
        return
    (_, _),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    x_test=(x_test.astype("float32")/255.0)[...,None]
    model=keras.models.load_model(BEST_MODEL)
    
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(x_test),size=n_samples,replace=False)
    imgs=x_test[idx]
    labs=y_test[idx]
    preds=model.predict(imgs,verbose=0).argmax(axis=1)
    
    cols=int(np.ceil(np.sqrt(n_samples)))
    rows=int(np.ceil(n_samples/cols))
    plt.figure(figsize=(2.8*cols,2.8*rows))
    for i in range(n_samples):
        plt.subplot(rows,cols,i+1)
        plt.imshow(imgs[i].squeeze(),cmap="gray")
        plt.title(f"P:{FASHION_CLASSES[preds[i]]}\nT:{FASHION_CLASSES[labs[i]]}",fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(ARTIFACT_DIR)
    out_path=os.path.join(ARTIFACT_DIR,"inference_grid.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved:",out_path)
    
####################################################################
# Function : run_baselines
# Inputs: None
# Outputs: Prints accuracy of LogReg,LinearSVC,RandomForest
# Description: Runs classical ML baselines on flattened Fashion-MNIST
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def run_baselines():
    x_train,y_train,x_test,y_test=load_flattened()
    logreg=Pipeline([
        ("Scaler",StandardScaler(with_mean=True)),
        ("clf",LogisticRegression(max_iter=200,n_jobs=-1))
    ])
    logreg.fit(x_train,y_train)
    print("LogReg acc:",accuracy_score(y_test,logreg.predict(x_test)))
    
    svm=Pipeline([
        ("scaler",StandardScaler(with_mean=True)),
        ("clf",LinearSVC(C=1.0,dual=True,max_iter=5000))
    ])
    svm.fit(x_train,y_train)
    print("LinearSVC acc:",accuracy_score(y_test,svm.predict(x_test)))
    
    rf=RandomForestClassifier(n_estimators=200,max_depth=None,n_jobs=-1,random_state=SEED)
    rf.fit(x_train,y_train)
    print("RandomForest acc:",accuracy_score(y_test,rf.predict(x_test)))

####################################################################
# Function : parse_args
# Inputs: None
# Outputs: Parsed CLI arguments
# Description: Defines command-line arguments for training,inference,baselines
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def parse_args():
    p=argparse.ArgumentParser(description="Fashion-MNIST Case Study")
    p.add_argument("--train",action="store_true",help="Train the CNN and save the artifacts.")
    p.add_argument("--infer",action="store_true",help="Generate an inference grid using saved model.")
    p.add_argument("--samples",type=int,default=9,help="Number of samples for inference grid.")
    p.add_argument("--baselines",action="store_true",help="Run classical ML baselines.")
    p.add_argument("--batch",type=int,default=128,help="Batch size for CNN training.")
    p.add_argument("--epochs",type=int,default=15,help="Number of epochs for CNN training.")
    p.add_argument("--lr",type=float,default=1e-3,help="Learning rate for Adam.")
    return p.parse_args()

####################################################################
# Function : main
# Inputs: None
# Outputs: None
# Description: Entry point. Parses CLI args,runs train/infer/baselines accordingly.
# Author: Yash Gorakshnath Andhale
# Date: 04/10/2025
####################################################################

def main():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    if args.train:
        train_and_evaluate(batch_size=args.batch,epochs=args.epochs,lr=args.lr)
        did_anything=True
    if args.infer:
        inference_grid(n_samples=args.samples)
        did_anything=True
    if args.baselines:
        run_baselines()
        did_anything=True
    if not did_anything:
        print("Nothing to do.Try:\n python fashion_mnist_case_study.py --train")
        
####################################################################
if __name__ =="__main__":
    main()
####################################################################
