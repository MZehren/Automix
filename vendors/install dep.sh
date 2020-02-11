#----base----
python -m pip install --upgrade pip
pip install numpy
pip install scipy
pip install cython
pip install matplotlib
pip install findiff

#----install madmom----
#might fail: https://github.com/SlapBot/stephanie-va/issues/8 (sudo apt-get install libportaudio2)
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio 
pip install pyaudio
pip install madmom

#----install librosa----
pip install librosa # ==0.5.1 NB < 6.0 needed for msaf 
apt-get install python-tk
apt-get install ffmpeg

#----install essentia----
pip install essentia

#---- install mir_eval----
pip install mir_eval

#---- install msaf----
pip install msaf

#----install graph_viz----
pip install graphviz

#----install IO downloader----
sudo -H pip install --upgrade youtube-dl

#----install madmom for drums detection----
#This is a branch of madmom, it doesn't contain the last update of the main project
#You have to install it in a different environment because it's not possible to install two versions of the same repository.
# Updated to python3 (python3 -m venv madmomDrumsEnv) doesn't work for the installation. for now keeping python2
virtualenv madmomDrumsEnv 

#install the dependencies. 
madmomDrumsEnv/bin/pip install numpy
madmomDrumsEnv/bin/pip install scipy
madmomDrumsEnv/bin/pip install cython
madmomDrumsEnv/bin/pip install nose
#might fail ?
sudo apt-get install python-dev
madmomDrumsEnv/bin/pip install pyaudio

#install from the source `: http://www.ifs.tuwien.ac.at/~vogl/models/mirex-17.zip`
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

#----install soundfile decoding library
pip install audioread

#----Vocal Melody Extraction
git clone https://github.com/s603122001/Vocal-Melody-Extraction.git
cd Vocal-Melody-Extraction/
rm -rf .git
echo "!!!! You have to download manually: https://drive.google.com/file/d/13kApyZ5lJEGE5CDwaeEuxVuw9sZy_xae/view !!!! (can't use wget with google drive")
echo "I put the models in the folder, added a setup.py to install the project"
pip install -e .
echo "then I changed the loading function in melodyExt from soundfile to librosa to work with mp3."
# Then add the dependencies
pip install tensorflow
pip install keras
pip install soundfile
pip install tqdm



#----other----
pip install mongoengine

#----Not used anymore----
#install jupyter
#python -m pip install jupyter

#install melodia
#move the melodia's file to a vamp folder
#pip install vamp

#install msa 
#pip install msaf
