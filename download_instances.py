"""
Download TSPLIB instances from GitHub mirror (official site is down)
"""
import urllib.request
import os

# GitHub mirror of TSPLIB instances
BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master/"

INSTANCES = {
    'eil51.tsp': 'eil51.tsp',
    'st70.tsp': 'st70.tsp',
    'pr107.tsp': 'pr107.tsp',
    'ch130.tsp': 'ch130.tsp',
    'a280.tsp': 'a280.tsp'
}

def download_instance(filename):
    """Download a TSP instance from GitHub mirror."""
    url = BASE_URL + filename
    print(f"Downloading {filename} from GitHub mirror...")

    try:
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')

        with open(filename, 'w') as f:
            f.write(content)

        print(f"  ✓ {filename} downloaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")
        return False

if __name__ == '__main__':
    print("Downloading TSPLIB instances from GitHub mirror...")
    print("=" * 60)

    success_count = 0
    for filename in INSTANCES.keys():
        if os.path.exists(filename):
            print(f"  • {filename} already exists, skipping")
            success_count += 1
        else:
            if download_instance(filename):
                success_count += 1

    print("=" * 60)
    print(f"Downloaded {success_count}/{len(INSTANCES)} instances")

    if success_count == len(INSTANCES):
        print("\n✓ All instances ready!")
        print("\nNext step: Run multi-instance experiments")
        print("  cd src")
        print("  python3 run_multi_instance.py")