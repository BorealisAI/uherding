# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

seeds=(1)
datasets=(domainnet) # tinyimagenet
features=(finetune)

model=RESNET18 # resnet18, DEIT_SMALL
kernel=rbf
delta=1.0
initial_size=0
unc_trans_fn=identity
unc_feature=simclr
tag=""
ent_weight=0.0
gamma=0.0
last_layer=fc
lr=-1.0
normalize=false
adaptive_delta=false
temp_scale=false
reg_pattern=uherding

is_integer() {
    [[ $1 =~ ^-?[0-9]+$ ]]
}

while getopts ":m:a:k:b:u:t:w:g:l:n:p:s:" flag; do
  case "${flag}" in
    m) model=${OPTARG};;
    a) al_method=${OPTARG};;
    k) kernel=${OPTARG};;
    b) set_budget=${OPTARG};;
    u) unc_feature=${OPTARG};;
    t) tag=${OPTARG};;
    w) ent_weight=${OPTARG};;
    g) gamma=${OPTARG};;
    l) last_layer=${OPTARG};;
    n) normalize=${OPTARG};;
    p) adaptive_delta=${OPTARG};;
    s) temp_scale=${OPTARG};;
    :)                                         # If expected argument omitted:
        echo "Error: -${OPTARG} requires an argument."
        exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
        exit_abnormal;;                          # Exit abnormally.
  esac
done

for seed in ${seeds[@]}; do
    for dataset_array in ${datasets[@]}; do
        # parse dataset
        dataset_array=(${dataset_array//_/ })
        dataset=${dataset_array[0]}
        imbal=${dataset_array[1]}

        if [[ -z "${imbal}" ]]
        then
            config_path="../configs/$dataset/al/${model}.yaml"
        else
            config_path="../configs/$dataset/al/${model}_IM.yaml"
        fi


        # determine budget size
        if [ $dataset == "cifar10" ]
        then
            if [[ -z "${set_budget}" ]]
            then
                budget=10
            else
                budget=${set_budget}
            fi
        elif [ $dataset == "cifar100" ]
        then
            if [[ -z "${set_budget}" ]]
            then
                budget=100
            else
                budget=${set_budget}
            fi
        elif [ $dataset == "imagenet" ]
        then
            if [[ -z "${set_budget}" ]]
            then
                budget=1000
            else
                budget=${set_budget}
            fi
        elif [ $dataset == "imagenet100" ]
        then
            if [[ -z "${set_budget}" ]]
            then
                budget=100
            else
                budget=${set_budget}
            fi
        elif [ $dataset == "tinyimagenet" ]
        then
            if [[ -z "${set_budget}" ]]
            then
                budget=200
            else
                budget=${set_budget}
            fi
        else
            if [[ -z "${set_budget}" ]]
            then
                budget=10
            else
                budget=${set_budget}
            fi
        fi

        if is_integer "$budget"; then
            initial_size=$budget
        else
            budget_array=(${budget//_/ })
            initial_size=${budget_array[0]}
        fi


        # determine delta for probcover and herding
        if [ $al_method == "probcover" ] 
        then
            if [ $dataset == "cifar10" ]
            then
                if [[ -z "${imbal}" ]]
                then
                    delta=0.55
                else
                    delta=0.55
                fi
            elif [ $dataset == "cifar100" ]
            then
                if [[ -z "${imbal}" ]]
                then
                    delta=0.5
                else
                    delta=0.5
                fi
            elif [ $dataset == "stl10" ]
            then
                delta=0.6
            elif [ $dataset == "imagenet" ] || [ $dataset == "imagenet100" ]
            then
                if [[ -z "${imbal}" ]]
                then
                    delta=0.55
                else
                    delta=0.55
                fi
            elif [ $dataset == "tinyimagenet" ]
            then
                if [[ -z "${imbal}" ]]
                then
                    delta=0.55
                else
                    delta=0.5
                fi
            fi
        fi

        # determine initial_size
        if [[ $al_method == "probcover" ]] || [[ $al_method == "select_al" ]] || [[ $al_method == "knn" ]] || [[ $al_method == "herding" ]] || [[ $al_method == "kherding" ]] || [[ $al_method == kcherding* ]] || [[ $al_method == uherding* ]] || [[ $al_method == "wherding" ]] || [[ $al_method == "typiclust" ]] || [[ $al_method == "kkmedoids" ]] || [[ $al_method == "kmedoids" ]] || [[ $al_method == "kmedoids" ]] || [[ $al_method == "kmeans" ]] || [[ $al_method == "pam" ]] || [[ $al_method == badge_herding* ]] || [[ $al_method == "activeft" ]]
        then
            initial_size=0
        fi

        for feature in ${features[@]}; do
            if [ $al_method == "probcover" ] && [ $dataset == "cifar10" ] && [ $feature == "scan" ]
            then
                delta=0.55
            elif [ $al_method == "probcover" ] && [ $dataset == "cifar10" ] && [ $feature == "dino" ]
            then
                delta=0.6
            fi
            echo "************************************************"
            echo "Tag: $tag"
            echo "Seed: $seed"
            echo "Dataset: ${dataset}${imbal}"
            echo "Model: ${model}"
            echo "Budget: $budget"
            echo "LR: $lr"
            echo "Init size: $initial_size"
            echo "Method: $al_method"
            echo "Feature: $feature"
            echo "Delta: $delta"
            echo "Config: $config_path"
            echo "Unc trans func: $unc_trans_fn"
            echo "Unc feature: $unc_feature"
            echo "Gamma: $gamma"
            echo "Ent weight: $ent_weight"
            echo "Last layer: $last_layer"
            echo "Normalize: $normalize"
            echo "Adaptive delta: $adaptive_delta"
            echo "Temperature scale: $temp_scale"
            echo "************************************************"

            if [[ "$feature" == "random" || "$feature" == "finetune" ]]
            then
                python train_al.py --cfg $config_path --al $al_method --feature $feature \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "simclr" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature simclr \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "simclr_nn" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature simclr \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features --nn \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "scan" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature scan \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "scan_nn" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature scan \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features --nn \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "dino" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature dino \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            elif [ $feature == "dino_nn" ]
            then
                python train_al.py --cfg $config_path --al $al_method --feature dino \
                    --exp-name auto --initial_size $initial_size --budget $budget \
                    --delta $delta --kernel $kernel --seed $seed --tag "$tag" --linear_from_features --nn \
                    --unc_trans_fn $unc_trans_fn --unc_feature $unc_feature \
                    --ent_weight $ent_weight --gamma $gamma --last_layer $last_layer --lr $lr \
                    --normalize $normalize --adaptive_delta $adaptive_delta --temp_scale $temp_scale
            fi
        done
    done
done

