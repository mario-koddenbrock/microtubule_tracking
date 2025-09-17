import os
import io
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
        if path_or_buf is None:
            return False
        # Check if the weight files are "large enough" (to make sure it's not just empty)
        if isinstance(path_or_buf, io.BytesIO):
            size = path_or_buf.getbuffer().nbytes
        else:
            size = os.path.getsize(path_or_buf)  # also gives bytes
        if size < 1_000:  # maybe too small to be legitimate
            print(f"\tCheckpoint file too small ({size} bytes)")
            return False
        else:
            print(f"Checkpoint file size: {size} bytes")
            return True

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
    print(f"Valid combinations for which weights exist:")
    for comb in valid:
        print(" ", comb)

    # Result of running this on September 17, 2025:
    # Valid combinations for which weights exist:
    # ('microtubules_tirf', 'fnet_attn', '32')
    # ('microtubules', 'dist', 'triang')
    # ('2d', 'dist', 'triang')


if __name__ == "__main__":
    main()
