# Automix

Automatic DJ-mixing of tracks

## Install

Clone or download the repository, then run from within the folder:
    
    pip install .

Or to keep the project editable, use:

    pip install . --editable

## Dependencies

Installing the project with pip should download all the dependencies except Richard Vogl's drums transcription:
This project is a fork of Madmom, and has to be installed it in a different environment to keep both libraries accessible.

    cd vendors
    # python3 -m venv madmomDrumsEnv doesn't work for the installation. for now keeping python2
    virtualenv madmomDrumsEnv 

    #install the dependencies. 
    madmomDrumsEnv/bin/pip install numpy
    madmomDrumsEnv/bin/pip install scipy
    madmomDrumsEnv/bin/pip install cython
    madmomDrumsEnv/bin/pip install nose
    #might fail ?
    sudo apt-get install python-dev
    madmomDrumsEnv/bin/pip install pyaudio

    #install from the source `http://www.ifs.tuwien.ac.at/~vogl/models/mirex-17.zip`
    wget http://www.ifs.tuwien.ac.at/~vogl/models/mirex-18.tar.gz
    tar -xvzf mirex-18.tar.gz
    cd madmom-0.16.dev0/
    # --user seems not available anymore
    ../madmomDrumsEnv/bin/python setup.py develop 
    #check if it's working
    cd ..
    madmomDrumsEnv/bin/python madmom-0.16.dev0/bin/DrumTranscriptor
    #clean everything
    rm mirex-18.tar.gz

## Usage
    from automix.model.classes.track import Track
    track = Track(path="path to audio file")
    cues = track.getCueIns()
    times = cues.times
    confidences = cues.values