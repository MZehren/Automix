import argparse

from automix import config
from automix.model.classes.track import Track


def main():
    # Load the tracks
    parser = argparse.ArgumentParser(description='Estimate the cue in points')
    parser.add_argument('folder', type=str, help="Path to the input folder containing tracks.")
    args = parser.parse_args()
    tracks = [Track(path=path) for path in config.getFolderFiles(args.folder)]

    # Estimate the cue points
    for t in tracks:
        cues = t.getCueIns()
        print(t, cues.values, cues.times)


if __name__ == '__main__':
    main()
