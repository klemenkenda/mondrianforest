#!/usr/bin/env bash
# script to download datasets for pre-process them

# download datasets
#for dir in 'usps' 'dna' 'satimage' 'letter'; do
#    echo ${dir}
#    cd ${dir}
#    ./commands.sh
#    cd ..
#done

# process libsvm datasets
python process_libsvm_datasets.py usps usps usps.t
python process_libsvm_datasets.py dna dna.scale.tr dna.scale.t   # ignoring validation files
python process_libsvm_datasets.py satimage satimage.scale.tr satimage.scale.t    # ignoring validation files
# ./process_libsvm_datasets.py letter letter.scale.tr letter.scale.t  # ignoring validation files
python process_libsvm_datasets.py letter letter.scale.tr letter.scale.t letter.scale.val # adding validation to training (to make comparisons with OnlineRF by Saffari et al.)
cd dna-61-120
python process_dna-61-120.py
cd ..
python convert_pickle_2_onlinerf_format.py dna-61-120
