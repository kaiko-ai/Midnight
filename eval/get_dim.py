import sys

def get_dim(name, check_concat=False):
    dim = 384
    if "h_optimus_0" in name.lower():
        dim = 1536
    elif "virchow2" in name.lower():
        dim = 1280
    elif "owkin_vits16_phikon" in name.lower() or "dino_vits16_phikon" in name.lower():
        dim = 768
    elif "vitl1" in name.lower():
        dim = 1024
    elif "vitb1" in name.lower():
        dim = 768
    elif "vitb8" in name.lower():
        dim = 768
    elif "vitg14" in name.lower():
        dim = 1536

    if check_concat and "concat" in name:
        return dim * 2
    return dim

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_dim.py <model_name> [<check_concat>]")
        exit(1)
    print(get_dim(sys.argv[1], int(sys.argv[2]) if len(sys.argv) >= 3 else False))
