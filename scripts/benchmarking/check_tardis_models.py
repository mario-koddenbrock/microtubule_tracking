import os
from tardis_em.utils.aws import get_all_version_aws, get_weights_aws
import itertools


def check_combination(network: str, subtype: str, dataset: str, version: int | None):
    """
    Try to fetch weights for the given combination.
    Returns True if succeeded (file exists or download succeeded), False otherwise.
    """
    try:
        # get_all_version_aws returns something like ["V_1", "V_2", ...] or empty list
        versions = get_all_version_aws(network, subtype, dataset)
        if not versions:
            # No versioned weights? Try version=None
            # version=None means latest if available
            path_or_buf = get_weights_aws(
                network=network, subtype=subtype, model=dataset, version=None
            )
            # If it returns a path (file on disk) or a buffer, then consider success
            return path_or_buf is not None
        else:
            # Try the latest version
            # Extract integer versions from names like "V_1", "V_2"
            ints = []
            for v in versions:
                try:
                    # remove prefix "V_"
                    iv = int(v.split("_")[1])
                    ints.append((iv, v))
                except Exception:
                    pass
            if not ints:
                # If version strings are weird, still try with version=None
                return (
                    get_weights_aws(network=network, subtype=subtype, model=dataset, version=None)
                    is not None
                )

            # pick highest integer version
            max_iv, max_vstr = max(ints, key=lambda x: x[0])
            # now try fetching that version
            path_or_buf = get_weights_aws(
                network=network, subtype=subtype, model=dataset, version=max_iv
            )
            return path_or_buf is not None

    except AssertionError as ae:
        # likely missing network or invalid model etc
        print(f"AssertionError for {network}, {subtype}, {dataset}, version {version}: {ae}")
        return False
    except Exception as e:
        print(f"Error for {network}, {subtype}, {dataset}, version {version}: {e}")
        return False


def main():
    ALL_NETWORKS = ["unet", "unet3plus", "fnet_attn", "dist"]
    ALL_SUBTYPES = ["16", "32", "64", "96", "128", "triang", "full"]
    # Focus on 2D and microtubules
    CNN_datasets = ["microtubules_2d", "microtubules_tirf"]
    DIST_datasets = ["microtubules", "2d"]  # only for network type "dist"
    datasets = CNN_datasets + DIST_datasets
    valid = []
    for dataset in datasets:
        for network, subtype in itertools.product(ALL_NETWORKS, ALL_SUBTYPES):
            # Ignore impossible ones
            if network == "dist" and dataset not in DIST_datasets:
                continue
            if network != "dist" and dataset in DIST_datasets:
                continue
            print(f"Testing network={network}, subtype={subtype}, dataset={dataset}...")
            ok = check_combination(network, subtype, dataset, version=None)
            print(f" → {'FOUND' if ok else 'missing'}")
            if ok:
                valid.append((dataset, network, subtype))
    print(f"Valid combinations:")
    for comb in valid:
        print(" ", comb)

    # Result of running this on September 17, 2025:
    # Valid combinations:
    # ('microtubules_2d', 'unet', '16')
    # ('microtubules_2d', 'unet', '32')
    # ('microtubules_2d', 'unet', '64')
    # ('microtubules_2d', 'unet', '96')
    # ('microtubules_2d', 'unet', '128')
    # ('microtubules_2d', 'unet', 'triang')
    # ('microtubules_2d', 'unet', 'full')
    # ('microtubules_2d', 'unet3plus', '16')
    # ('microtubules_2d', 'unet3plus', '32')
    # ('microtubules_2d', 'unet3plus', '64')
    # ('microtubules_2d', 'unet3plus', '96')
    # ('microtubules_2d', 'unet3plus', '128')
    # ('microtubules_2d', 'unet3plus', 'triang')
    # ('microtubules_2d', 'unet3plus', 'full')
    # ('microtubules_2d', 'fnet_attn', '16')
    # ('microtubules_2d', 'fnet_attn', '32')
    # ('microtubules_2d', 'fnet_attn', '64')
    # ('microtubules_2d', 'fnet_attn', '96')
    # ('microtubules_2d', 'fnet_attn', '128')
    # ('microtubules_2d', 'fnet_attn', 'triang')
    # ('microtubules_2d', 'fnet_attn', 'full')
    # ('microtubules_tirf', 'unet', '16')
    # ('microtubules_tirf', 'unet', '32')
    # ('microtubules_tirf', 'unet', '64')
    # ('microtubules_tirf', 'unet', '96')
    # ('microtubules_tirf', 'unet', '128')
    # ('microtubules_tirf', 'unet', 'triang')
    # ('microtubules_tirf', 'unet', 'full')
    # ('microtubules_tirf', 'unet3plus', '16')
    # ('microtubules_tirf', 'unet3plus', '32')
    # ('microtubules_tirf', 'unet3plus', '64')
    # ('microtubules_tirf', 'unet3plus', '96')
    # ('microtubules_tirf', 'unet3plus', '128')
    # ('microtubules_tirf', 'unet3plus', 'triang')
    # ('microtubules_tirf', 'unet3plus', 'full')
    # ('microtubules_tirf', 'fnet_attn', '16')
    # ('microtubules_tirf', 'fnet_attn', '32')
    # ('microtubules_tirf', 'fnet_attn', '64')
    # ('microtubules_tirf', 'fnet_attn', '96')
    # ('microtubules_tirf', 'fnet_attn', '128')
    # ('microtubules_tirf', 'fnet_attn', 'triang')
    # ('microtubules_tirf', 'fnet_attn', 'full')
    # ('microtubules', 'dist', '16')
    # ('microtubules', 'dist', '32')
    # ('microtubules', 'dist', '64')
    # ('microtubules', 'dist', '96')
    # ('microtubules', 'dist', '128')
    # ('microtubules', 'dist', 'triang')
    # ('microtubules', 'dist', 'full')
    # ('2d', 'dist', '16')
    # ('2d', 'dist', '32')
    # ('2d', 'dist', '64')
    # ('2d', 'dist', '96')
    # ('2d', 'dist', '128')
    # ('2d', 'dist', 'triang')
    # ('2d', 'dist', 'full')


if __name__ == "__main__":
    main()
