import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"


import warnings
warnings.filterwarnings("ignore")


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)


import torch
import torch.nn as nn


from torch.utils.data import DataLoader


from tqdm import tqdm


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


from app.modules.fusion.FusionDatasetCached import (
    FusionDatasetCached
)


from app.modules.fusion.fusion_model import (
    FusionModel
)



####################################################
# DEVICE
####################################################

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"

)


print()
print("=" * 60)
print("DEVICE")
print("=" * 60)

print(DEVICE)

print("=" * 60)
print()



####################################################
# MODEL PATHS
####################################################


IMAGE_MODEL_PATH = (

    "models/image_model/"
    "Hybrid_Swin_EffNet_best1.pth"

)


VIDEO_MODEL_PATH = (

    "models/video_model/"
    "video_model.pth"

)


AUDIO_MODEL_PATH = (

    "models/audio_model/"
    "audio_model.pth"

)


LIP_MODEL_PATH = (

    "models/lip_sync_model/"
    "sync_model_final_best.pth"

)



####################################################
# CACHE
####################################################

CACHE_DIR = "fusion_cache"



####################################################
# SAVE DIRECTORY
####################################################

SAVE_DIR = "models/fusion_model"


os.makedirs(

    SAVE_DIR,

    exist_ok=True

)



####################################################
# TRAINING SETTINGS
####################################################

BATCH_SIZE = 4


EPOCHS_STAGE1 = 10


EPOCHS_STAGE2 = 10


EARLY_STOPPING_PATIENCE = 5



####################################################
# DATASETS
####################################################

print()
print("=" * 60)
print("Loading Fusion Cache")
print("=" * 60)



train_dataset = FusionDatasetCached(

    CACHE_DIR,

    split="train"

)



val_dataset = FusionDatasetCached(

    CACHE_DIR,

    split="val"

)



print("=" * 60)
print()



####################################################
# DATASET BALANCE CHECK
####################################################

real_count = 0
fake_count = 0


for sample in train_dataset.samples:

    if sample["label"] == 0:

        real_count += 1

    else:

        fake_count += 1



print()

print("=" * 60)

print("DATASET DISTRIBUTION")

print("=" * 60)

print(
    "Real samples:",
    real_count
)

print(
    "Fake samples:",
    fake_count
)

print(
    "Total:",
    len(train_dataset)
)

print("=" * 60)

print()



####################################################
# DATALOADERS
# BALANCED DATASET -> NORMAL SHUFFLE
####################################################


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True,

    drop_last=True

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)



print()

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Val batches:",
    len(val_loader)
)

print()



####################################################
# LOAD FUSION MODEL
####################################################


print("=" * 60)

print("Loading Fusion Model")

print("=" * 60)



model = FusionModel(

    image_model_path=IMAGE_MODEL_PATH,

    video_model_path=VIDEO_MODEL_PATH,

    audio_model_path=AUDIO_MODEL_PATH,

    lip_model_path=LIP_MODEL_PATH

)



model.to(DEVICE)



print()

print(
    "Total Trainable Parameters:",
    model.count_trainable_parameters()
)

print()



####################################################
# LOSS
# NO CLASS WEIGHTS
####################################################


criterion = nn.CrossEntropyLoss(

    label_smoothing=0.05

)



####################################################
# STAGE 1
# TRAIN ONLY FUSION LAYERS
####################################################


model.freeze_feature_extractors()



optimizer = torch.optim.AdamW(

    filter(

        lambda p: p.requires_grad,

        model.parameters()

    ),

    lr=3e-4,

    weight_decay=1e-4

)



####################################################
# LR SCHEDULER
####################################################


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS_STAGE1

)



####################################################
# AMP
####################################################


scaler = torch.cuda.amp.GradScaler(

    enabled=(DEVICE.type == "cuda")

)



####################################################
# HISTORY
####################################################


history = {

    "train_loss": [],

    "train_acc": [],

    "val_loss": [],

    "val_acc": [],

    "precision": [],

    "recall": [],

    "f1": [],

    "auc": []

}



####################################################
# CHECKPOINT VARIABLES
####################################################


best_f1 = 0.0


epochs_without_improvement = 0



print()

print("=" * 60)

print("SETUP COMPLETE")

print("=" * 60)

print()
####################################################
# TRAIN ONE EPOCH
####################################################


def train_one_epoch():


    model.train()


    running_loss = 0.0


    all_predictions = []

    all_labels = []



    progress = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )



    for batch in progress:



        ################################################
        # LOAD DATA
        ################################################


        image = batch["image"].to(

            DEVICE,

            non_blocking=True

        )


        video = batch["video"].to(

            DEVICE,

            non_blocking=True

        )


        audio = batch["audio"].to(

            DEVICE,

            non_blocking=True

        )


        lip = batch["lip"].to(

            DEVICE,

            non_blocking=True

        )


        labels = batch["label"].to(

            DEVICE,

            non_blocking=True

        )



        ################################################
        # ZERO GRADIENT
        ################################################


        optimizer.zero_grad(

            set_to_none=True

        )



        ################################################
        # FORWARD PASS
        ################################################


        with torch.cuda.amp.autocast(

            enabled=(DEVICE.type == "cuda")

        ):


            outputs = model(

                image,

                video,

                audio,

                lip

            )



            loss = criterion(

                outputs,

                labels

            )



        ################################################
        # BACKPROPAGATION
        ################################################


        scaler.scale(

            loss

        ).backward()



        ################################################
        # GRADIENT CLIPPING
        ################################################


        scaler.unscale_(

            optimizer

        )



        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )



        ################################################
        # OPTIMIZER STEP
        ################################################


        scaler.step(

            optimizer

        )


        scaler.update()



        ################################################
        # METRICS
        ################################################


        running_loss += loss.item()



        predictions = torch.argmax(

            outputs,

            dim=1

        )



        all_predictions.extend(

            predictions.detach()

            .cpu()

            .numpy()

        )


        all_labels.extend(

            labels.detach()

            .cpu()

            .numpy()

        )



        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )



    ####################################################
    # EPOCH RESULTS
    ####################################################


    epoch_loss = (

        running_loss

        /

        len(train_loader)

    )



    epoch_accuracy = (

        accuracy_score(

            all_labels,

            all_predictions

        )

        *

        100

    )



    return (

        epoch_loss,

        epoch_accuracy

    )
####################################################
# VALIDATION
####################################################


def validate():


    model.eval()


    running_loss = 0.0


    all_predictions = []

    all_labels = []

    all_probabilities = []



    with torch.no_grad():


        progress = tqdm(

            val_loader,

            desc="Validation",

            leave=False

        )



        for batch in progress:



            ################################################
            # LOAD DATA
            ################################################


            image = batch["image"].to(

                DEVICE

            )


            video = batch["video"].to(

                DEVICE

            )


            audio = batch["audio"].to(

                DEVICE

            )


            lip = batch["lip"].to(

                DEVICE

            )


            labels = batch["label"].to(

                DEVICE

            )



            ################################################
            # FORWARD
            ################################################


            outputs = model(

                image,

                video,

                audio,

                lip

            )



            loss = criterion(

                outputs,

                labels

            )



            running_loss += loss.item()



            ################################################
            # PROBABILITY
            ################################################


            probabilities = torch.softmax(

                outputs,

                dim=1

            )[:,1]



            predictions = torch.argmax(

                outputs,

                dim=1

            )



            ################################################
            # STORE RESULTS
            ################################################


            all_predictions.extend(

                predictions.cpu()

                .numpy()

            )


            all_labels.extend(

                labels.cpu()

                .numpy()

            )


            all_probabilities.extend(

                probabilities.cpu()

                .numpy()

            )




    ####################################################
    # CALCULATE METRICS
    ####################################################


    val_loss = (

        running_loss

        /

        len(val_loader)

    )



    accuracy = (

        accuracy_score(

            all_labels,

            all_predictions

        )

        *

        100

    )



    precision = precision_score(

        all_labels,

        all_predictions,

        zero_division=0

    )



    recall = recall_score(

        all_labels,

        all_predictions,

        zero_division=0

    )



    f1 = f1_score(

        all_labels,

        all_predictions,

        zero_division=0

    )



    try:


        auc = roc_auc_score(

            all_labels,

            all_probabilities

        )


    except:


        auc = 0.0




    cm = confusion_matrix(

        all_labels,

        all_predictions

    )




    ####################################################
    # PRINT RESULTS
    ####################################################


    print()

    print("=" * 60)

    print("VALIDATION RESULTS")

    print("=" * 60)



    print(

        f"Loss      : {val_loss:.4f}"

    )


    print(

        f"Accuracy  : {accuracy:.2f}%"

    )


    print(

        f"Precision : {precision:.4f}"

    )


    print(

        f"Recall    : {recall:.4f}"

    )


    print(

        f"F1 Score  : {f1:.4f}"

    )


    print(

        f"ROC-AUC   : {auc:.4f}"

    )


    print()

    print("Confusion Matrix")

    print(cm)

    print("=" * 60)

    print()



    return (

        val_loss,

        accuracy,

        precision,

        recall,

        f1,

        auc

    )
####################################################
# STAGE 1
# ONLY FUSION LAYERS TRAIN
####################################################


print()

print("=" * 60)

print("STAGE 1 : TRAINING FUSION LAYERS")

print("=" * 60)

print()



for epoch in range(EPOCHS_STAGE1):


    print()

    print("-" * 60)

    print(

        f"Epoch {epoch+1}/{EPOCHS_STAGE1}"

    )

    print("-" * 60)



    ################################################
    # TRAIN
    ################################################


    train_loss, train_acc = train_one_epoch()



    ################################################
    # VALIDATION
    ################################################


    (
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        auc

    ) = validate()



    ################################################
    # LR UPDATE
    ################################################


    scheduler.step()



    ################################################
    # SAVE HISTORY
    ################################################


    history["train_loss"].append(

        train_loss

    )


    history["train_acc"].append(

        train_acc

    )


    history["val_loss"].append(

        val_loss

    )


    history["val_acc"].append(

        val_acc

    )


    history["precision"].append(

        precision

    )


    history["recall"].append(

        recall

    )


    history["f1"].append(

        f1

    )


    history["auc"].append(

        auc

    )



    ################################################
    # PRINT SUMMARY
    ################################################


    print()

    print(

        f"Train Loss : {train_loss:.4f}"

    )

    print(

        f"Train Acc  : {train_acc:.2f}%"

    )


    print()

    print(

        f"Val Loss   : {val_loss:.4f}"

    )

    print(

        f"Val Acc    : {val_acc:.2f}%"

    )

    print(

        f"F1 Score   : {f1:.4f}"

    )

    print(

        f"AUC        : {auc:.4f}"

    )




    ################################################
    # SAVE BEST STAGE 1 MODEL
    ################################################


    if f1 > best_f1:


        best_f1 = f1


        epochs_without_improvement = 0



        torch.save(

            model.state_dict(),

            os.path.join(

                SAVE_DIR,

                "best_fusion_stage1.pth"

            )

        )



        print()

        print(

            "✓ Saved Best Stage 1 Fusion Model"

        )

        print(

            "Best F1:",

            best_f1

        )



    else:


        epochs_without_improvement += 1



        print()

        print(

            "No improvement"

        )


        print(

            "Patience:",

            epochs_without_improvement,

            "/",

            EARLY_STOPPING_PATIENCE

        )




    ################################################
    # EARLY STOPPING
    ################################################


    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:


        print()

        print(

            "Early stopping Stage 1"

        )

        break





print()

print("=" * 60)

print("STAGE 1 COMPLETED")

print("=" * 60)

print()


print(

    "Best Stage 1 F1:",

    best_f1

)

print()
####################################################
# STAGE 2
# FULL MODEL FINE TUNING
####################################################


print()

print("=" * 60)

print("STAGE 2 : FULL MODEL FINE TUNING")

print("=" * 60)

print()



####################################################
# LOAD BEST STAGE 1 CHECKPOINT
####################################################


stage1_path = os.path.join(

    SAVE_DIR,

    "best_fusion_stage1.pth"

)



if os.path.exists(stage1_path):


    model.load_state_dict(

        torch.load(

            stage1_path,

            map_location=DEVICE

        )

    )


    print()

    print(

        "Loaded best Stage 1 checkpoint"

    )



else:


    print()

    print(

        "Stage 1 checkpoint not found"

    )




####################################################
# UNFREEZE BACKBONES
####################################################


model.unfreeze_backbones()



print()

print(

    "Trainable parameters after unfreeze:",

    model.count_trainable_parameters()

)

print()



####################################################
# NEW OPTIMIZER
####################################################


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-5,

    weight_decay=1e-5

)



####################################################
# NEW SCHEDULER
####################################################


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS_STAGE2

)



####################################################
# RESET BEST TRACKING
####################################################


best_stage2_f1 = 0.0


epochs_without_improvement = 0




####################################################
# STAGE 2 LOOP
####################################################


for epoch in range(EPOCHS_STAGE2):


    print()

    print("-" * 60)

    print(

        f"Fine Tune Epoch {epoch+1}/{EPOCHS_STAGE2}"

    )

    print("-" * 60)



    ################################################
    # TRAIN
    ################################################


    train_loss, train_acc = train_one_epoch()



    ################################################
    # VALIDATION
    ################################################


    (
        val_loss,
        val_acc,
        precision,
        recall,
        f1,
        auc

    ) = validate()




    ################################################
    # LR UPDATE
    ################################################


    scheduler.step()



    ################################################
    # PRINT RESULTS
    ################################################


    print()

    print(

        f"Train Loss : {train_loss:.4f}"

    )


    print(

        f"Train Acc  : {train_acc:.2f}%"

    )


    print()

    print(

        f"Val Loss   : {val_loss:.4f}"

    )


    print(

        f"Val Acc    : {val_acc:.2f}%"

    )


    print(

        f"F1 Score   : {f1:.4f}"

    )


    print(

        f"AUC        : {auc:.4f}"

    )




    ################################################
    # SAVE BEST FINAL MODEL
    ################################################


    if f1 > best_stage2_f1:


        best_stage2_f1 = f1


        epochs_without_improvement = 0



        save_path = os.path.join(

            SAVE_DIR,

            "best_fusion_model.pth"

        )



        torch.save(

            model.state_dict(),

            save_path

        )



        print()

        print(

            "✓ Saved Best Fusion Model"

        )


        print(

            "Best F1:",

            best_stage2_f1

        )



    else:


        epochs_without_improvement += 1


        print()

        print(

            "No improvement"

        )


        print(

            "Patience:",

            epochs_without_improvement,

            "/",

            EARLY_STOPPING_PATIENCE

        )





    ################################################
    # EARLY STOPPING
    ################################################


    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:


        print()

        print(

            "Early stopping Stage 2"

        )

        break





####################################################
# TRAINING COMPLETE
####################################################


print()

print("=" * 60)

print("FUSION TRAINING COMPLETE")

print("=" * 60)

print()



print(

    "Best Final F1:",

    best_stage2_f1

)


print()



print(

    "Saved model:"

)


print(

    os.path.join(

        SAVE_DIR,

        "best_fusion_model.pth"

    )

)

print()