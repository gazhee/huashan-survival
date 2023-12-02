import argparse

def parse_opts_mr_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input', type=str, help='Input file path')
    parser.add_argument('--mr_dir', default=['G:/glioma/final_data/huashan_MR/npy_file/TCGA_npy',
                                             'G:/glioma/final_data/huashan_MR/npy_file/glioma_huashan_npy'], 
                                             type = str,help='Choose a mr dir')
    parser.add_argument('--path_dir', default= ['G:/glioma/final_data/huashan_MR/pathology/TCGA',
                                                'G:/glioma/final_data/huashan_MR/pathology/huashan'], 
                                                type = str, help = 'select pathology dir')
    parser.add_argument('--csv_dir', default= ['G:/glioma/final_data/glioma_survive/TCGA.csv',
                                               'G:/glioma/final_data/glioma_survive/huashan.csv'], 
                                               type = str, help = 'select csv file')
    parser.add_argument('--pth_dir', default=['G:/glioma/final_data/glioma_survive/mr_pathology/survive_cnn_min_loss_path_and_mr.pth',
                                                          'G:/glioma/final_data/glioma_survive/mr_pathology/survive_cnn_min_loss_path_and_mr_huashan.pth'], 
                                                          type=str, help='select pth dir in MR and Pathology')
    parser.add_argument('--save_csv_dir', default=[['G:/glioma/final_data/glioma_survive/only_path/test_TCGA.csv',
                                                    'G:/glioma/final_data/glioma_survive/mr_pathology/train_TCGA.csv',
                                                    'G:/glioma/final_data/glioma_survive/mr_pathology/ex_test_huashan.csv',
                                                    'G:/glioma/final_data/glioma_survive/mr_pathology/ex_train_huashan.csv'],
                                                    ['G:/glioma/final_data/glioma_survive/mr_pathology/test_huashan.csv',
                                                     'G:/glioma/final_data/MR_csv/glioma_survive/mr_pathology/train_huashan.csv',
                                                     'G:/glioma/final_data/glioma_survivemr_pathology/ex_test_TCGA.csv',
                                                     'G:/glioma/final_data/glioma_survive/mr_pathology/ex_train_TCGA.csv']],
                                                     type=str, help='save_csv_dir')
    # parser.add_argument('--model', default='', type=str, help='Model file path')
    # parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    # parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    # parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    # parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    # parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    # parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    # parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    # parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    # parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    # parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    # parser.set_defaults(verbose=False)
    # parser.add_argument('--verbose', action='store_true', help='')
    # parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    opt = parse_opts_mr_path()
    print(opt.mr_dir)
