import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    args = parser.parse_args()
    print(args)

    # The data folder contains a subfolder for each teacher dataset
    # Each subfolder is assumed to contain 'train.tsv', 'dev.tsv', 'train_dev.tsv' and 'test.tsv' file.
    teachers_data_folders = os.listdir(args.data_path)
    modes = ['train.tsv', 'dev.tsv', 'train_dev.tsv', 'test.tsv']

    for folder in teachers_data_folders:
            os.makedirs(os.path.join(args.data_path, 'GLOBAL', folder))

    for mode in modes:
        for i in range(len(teachers_data_folders)):
            fout_path = os.path.join(args.data_path, "GLOBAL", teachers_data_folders[i], mode)
            fout = open(fout_path, "wt")
            print("Generating file: ", fout_path)

            for j in range(len(teachers_data_folders)):
                fin = open(os.path.join(args.data_path, teachers_data_folders[j], mode), "rt")

                if i == j:
                    for line in fin:
                        fout.write(line)
                else:
                    for line in fin:
                        replace_B = line.replace('\tB', '\tO')
                        replace_I = line.replace('\tI', '\tO')

                        if replace_B != line:
                            fout.write(replace_B)
                        elif replace_I != line:
                            fout.write(replace_I)
                        else:
                            fout.write(line)

                fin.close()
            fout.close()

    os.makedirs(os.path.join(args.data_path, 'GLOBAL', 'Student'))

    for mode in modes:
        fout_path = os.path.join(args.data_path, "GLOBAL", 'Student', mode)
        fout = open(fout_path, "wt")
        print("Generating file: ", fout_path)

        for i in range(len(teachers_data_folders)):
            fin = open(os.path.join(args.data_path, teachers_data_folders[i], mode), "rt")
            for line in fin:
                replace_B = line.replace('\tB', '\tB-'+teachers_data_folders[i])
                replace_I = line.replace('\tI', '\tI-'+teachers_data_folders[i])

                if replace_B != line:
                    fout.write(replace_B)
                elif replace_I != line:
                    fout.write(replace_I)
                else:
                    fout.write(line)
            fin.close()
        fout.close()

if __name__ == '__main__':
    main()