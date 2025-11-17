filepath="../shared/datasets/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas"
config_name="config_1"

python process_data.py \
    --filepath $filepath \
    --data_config $config_name \