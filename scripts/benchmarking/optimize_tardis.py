import os
from tardis_em.utils.aws import get_all_version_aws, get_weights_aws
import itertools


def check_combination(network: str, subtype: str, model: str, version: int | None):
    """
    Try to fetch weights for the given combination.
    Returns True if succeeded (file exists or download succeeded), False otherwise.
    """
    try:
        # get_all_version_aws returns something like ["V_1", "V_2", ...] or empty list
        versions = get_all_version_aws(network, subtype, model)
        if not versions:
            # No versioned weights? Try version=None
            # version=None means latest if available
            path_or_buf = get_weights_aws(network=network, subtype=subtype, model=model, version=None)
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
                return get_weights_aws(network=network, subtype=subtype, model=model, version=None) is not None

            # pick highest integer version
            max_iv, max_vstr = max(ints, key=lambda x: x[0])
            # now try fetching that version
            path_or_buf = get_weights_aws(network=network, subtype=subtype, model=model, version=max_iv)
            return path_or_buf is not None

    except AssertionError as ae:
        # likely missing network or invalid model etc
        print(f"AssertionError for {network}, {subtype}, {model}, version {version}: {ae}")
        return False
    except Exception as e:
        print(f"Error for {network}, {subtype}, {model}, version {version}: {e}")
        return False


def main():
    ALL_MODELS = ["unet", "unet3plus", "fnet_attn", "dist"]
    ALL_SUBTYPE = ["16", "32", "64", "96", "128", "triang", "full"]
    # Only for microtubules_2d
    model = "microtubules_2d"
    valid = []
    for network, subtype in itertools.product(ALL_MODELS, ALL_SUBTYPE):
        print(f"Testing network={network}, subtype={subtype}, model={model} ...")
        ok = check_combination(network, subtype, model, version=None)
        print(f" → {'FOUND' if ok else 'missing'}")
        if ok:
            valid.append((network, subtype))
    print("Valid combinations (network, subtype) for microtubules_2d:")
    for comb in valid:
        print(" ", comb)


if __name__ == "__main__":
    main()
