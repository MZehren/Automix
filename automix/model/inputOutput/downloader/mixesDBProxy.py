"""
Enables us to download from mixesdb.com
"""
import urllib2
import re

from model.inputOutput import reaperProxy, downloaders
from model.classes.deck import Deck


def getMixesDBMix(url, outputFolder):
    """
    create a reaper file in the outputfolder from the mix.
    the url should be https://www.mixesdb.com/w/2018-05-12_-_Camelphat_-_Essential_Mix
    """
    # parse the url
    page = urllib2.urlopen(url).read()

    tracks = []
    for ytId in re.findall("data-youtubeid=\\\\\"(.{11})\\\\\"", page):
        tracks.append(downloaders.downloadYoutube(ytId, outputFolder))
        tracks[-1].position = 101

    with open("mix.RPP", 'w') as outfile:
        outfile.write(
            reaperProxy.getReaperProject(decks=[Deck(tracks=tracks)]))


# getMixesDBMix("https://www.mixesdb.com/w/2018-05-12_-_Camelphat_-_Essential_Mix",
#   "annotations/mixes/House/")
