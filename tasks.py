from invoke import task


@task
def merge_lvis_objaverse_classes(ctx, lvis_train="../../datasets/coco/annotations/lvis_v1_train.json"):
    import prior
    import json

    objaverse_data = prior.load_dataset("objaverse-lvis")
    objaverse_classes = set(list(objaverse_data.keys()))
    print(len(objaverse_classes), "classes in objaverse")

    with open(lvis_train, "r") as f:
        lvis_data = json.load(f)
    lvis_classes = set([cat["name"] for cat in lvis_data["categories"]])
    print(len(lvis_classes), "classes in lvis")

    print(len(objaverse_classes | lvis_classes), "combined classes")
    print("new class:", objaverse_classes - lvis_classes)

    print("DONE")


@task
def load_objaverse(ctx, lvis_train="../../datasets/coco/annotations/lvis_v1_train.json"):
    from mmdet.datasets.objaverse import ObjaverseDataset
    # TODO: many missing components in call to Dataset - it fails after having called  load_annotations.
    dataset = ObjaverseDataset("../../datasets/coco/annotations/lvis_v1_train.json", [])
    print("DONE")
