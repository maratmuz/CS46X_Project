from datasets import load_dataset, get_dataset_config_names


def pgb_gene_exp(streaming: bool = False):
    """
    Load all gene_exp configs from InstaDeepAI/plant-genomic-benchmark.

    Args:
        streaming (bool): Whether to stream the datasets instead of downloading locally.

    Returns:
        dict: {species: DatasetDict}, e.g. {'glycine_max': DatasetDict(...), ...}
    """
    DATASET_NAME = "InstaDeepAI/plant-genomic-benchmark"
    CONFIG_PREFIX = "gene_exp."

    # get all configs from HF
    configs = get_dataset_config_names(DATASET_NAME)

    # filter for gene_exp configs
    gene_exp_configs = [c for c in configs if c.startswith(CONFIG_PREFIX)]

    datasets_dict = {}
    for config in gene_exp_configs:
        species = config[len(CONFIG_PREFIX):]
        datasets_dict[species] = load_dataset(
            DATASET_NAME,
            config,
            streaming=streaming,
        )

    return datasets_dict