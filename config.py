

class Config():

    #
    raw_dir = 'data/RawImage'
    annotation_dir = 'data/AnnotationsByMD'
    processing_dir = "data/cephalometric/"

    # train setting
    img_dir = 'data/cephalometric/768_768/TrainingData'
    gt_dir = 'data/cephalometric/768_768/gt'
    test1_dir = 'data/cephalometric/768_768/Test1Data'
    test2_dir = 'data/cephalometric/768_768/Test2Data'
    GPU = 3

    # parameter setting
    physical_factor = 0.1  #像素距离
    alpha = 40
    height = 768
    width = 768
    sigma = 12.5
    num_classes = 19
    num_epochs = 30
    lr = 1e-4


    # model parameters
    mlp_ratio = 4.
    drop_rate = 0.1
    drop_path_rate = 0.1 #0.3
    embed_dim = 128
    depths = [2, 2, 18, 2]
    window_size = 12
    head = 4
    num_heads = [head, head * 2, head * 4, head * 8]

    # save files

    save_model_path = 'outputs/model/'
    save_results_path = 'outputs/results/'

    save_huge_path = 'outputs/huge_MRE/'
    reference_path = 'outputs/reference/'

