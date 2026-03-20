LOCAL_CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
]

GLOBAL_CLASS_NAMES = [
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

NO_FINDING_CLASS_NAME = "No finding"
BACKGROUND_CLASS_NAME = "__background__"

CLASS_TO_LABEL = {class_name: idx for idx, class_name in enumerate(LOCAL_CLASS_NAMES, start=1)}
LABEL_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_LABEL.items()}
NUM_CLASSES = len(LOCAL_CLASS_NAMES) + 1
