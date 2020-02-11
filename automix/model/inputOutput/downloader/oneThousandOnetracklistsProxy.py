"""
Module for downloading tracklist from https://www.1001tracklists.com/
"""
import json
import logging
import re
import time
from urllib.request import urlopen

from bs4 import BeautifulSoup

from automix.model.classes.track import Track
from automix.model.inputOutput.serializer import DBSerializer


# from automix.model.inputOutput.downloader import downloadSo, downloadYoutube
class Container(dict):
    """
    Dictionnary with a add method not removing values
    # TODO replace with a default list dictionnary
    """

    def add(self, key, value):
        if key in self:
            if not isinstance(self[key], list):
                self[key] = [self[key]]
            self[key].append(value)

        else:
            self[key] = value


def scrapOverview(rangeToParse=range(0, 100), sleep=6):
    """
    Parse the page containing all the mixes
    """
    db = DBSerializer()
    for i in rangeToParse:
        logging.info("page :" + str(i))
        url = "https://www.1001tracklists.com/index" + str(i) + ".html"
        page = urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        for soupTracklist in soup.findAll("div", {"class": "tlLink"}):
            tlUrl = "https://www.1001tracklists.com" + soupTracklist.find("a")["href"]
            if db.exist({"_id": tlUrl}):
                logging.info(tlUrl + " already exist")
                continue

            mix = scrapTrackList(url=tlUrl)

            try:
                id = tlUrl #TODO: find a better ID, the url is based on the name which is subject to change
                db.insert(mix, id)
                logging.info("   - " + id[-20:] + " inserted")
            except Exception as e:
                logging.info("   - " + id[-20:] + "not inserted")
                logging.warn(e)

            time.sleep(sleep)


def scrapTrackList(url="https://www.1001tracklists.com/tracklist/20gg7q7t/wankelmut-1live-dj-session-2019-04-20.html"):
    """
    Parse the page containing the whole tracklist
    """
    # init
    page = urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')

    # Parse the mix data
    # TODO: add date of recording instead of date of publication
    mix = scrapMeta(soup.find("div", {"itemtype": "http://schema.org/MusicPlaylist"}))
    mix.add("medias", scrapMediaMix(soup.find("div", {"id": "mediaItems"})))

    # Parse the tracklist
    tracks = []
    for soupTrack in soup.findAll("div", {"itemtype": "http://schema.org/MusicRecording"}):
        if "tgHid" in soupTrack.parent.parent["class"]:
            continue  # TODO: do we remove the hiden tracks ?
        track = Container()
        for soupTrackMeta in soupTrack.findAll("meta"):
            track.add(soupTrackMeta["itemprop"], soupTrackMeta["content"])

        track.add("medias", scrapMediaTrack(soupTrack))

        order = soupTrack.parent.parent.find("td", {"class": "left"}).findAll("span")
        if len(order):
            track.add("order", order[-1].text)
            if track["order"] == "w/" and len(tracks):
                track["order"] = tracks[-1]["order"]
            try:
                track["order"] = int(track["order"])
            except ValueError:
                track["order"] = ""
        else:
            # Track has been likely removed
            continue

        track.add("position", soupTrack.parent.parent.find("td", {"class": "left"}).find("div").text)

        tracks.append(track)
        # for all tracks in the track list:
        # mixedTracks = []
        # playerSelector = CSSSelector('div[class="s32"]')
        # itemSelector = CSSSelector('table.tl tr.tlpItem')
        # playSelector = CSSSelector('div[title="play position"]')
        # nameSelector = CSSSelector('meta[itemprop="name"]')

    mix.add("tracks", tracks)
    return mix


def scrapMeta(soup):
    """
    return the metadata from any element inside soup:
    Look for: <meta itemprop="bla" content="blabla" />
    """
    container = Container()
    for soupMeta in soup.findAll("meta"):
        prop = soupMeta["itemprop"]
        value = soupMeta["content"]
        container.add(prop, value)
    return container


def scrapMediaMix(soup):
    """
        Return the media links from a mix (different than from a track)
        Soup should be a "mediaItemsTop"
        calls get_medialink.php?idMedia
    """
    medias = []
    for soupDiv in soup.findAll("div"):
        try:
            if soupDiv.has_attr("data-idmedia"):
                requestUrlUrl = "https://www.1001tracklists.com/ajax/get_medialink.php?idMedia=" + soupDiv[
                    "data-idmedia"] + "&dontHide=true&showInline=true"
                jsonResult = json.loads(urlopen(requestUrlUrl).read())["data"][0]
                medias.append(jsonResult)
        except Exception as e:
            logging.warn(e)
    return medias


def scrapMediaTrack(soup):
    """
    Scrap the media link for Youtube, soundcloud, etc, 
    input: soup output from a track element in a playlist
    TODO: calls get_medialink.php?idObject=...&idItem=... in another function
    """
    onClickContent = soup.parent.find("div", {"class": "addMedia"})
    # If it's a track, then the media should be in a addMedia div
    try:
        idItem = re.search("idItem: ([0-9]+)", str(onClickContent)).group(1)
        idObject = (re.search("idObject: ([0-9]+)", str(onClickContent)).group(1))
        return {"idItem": idItem, "idObject": idObject}
    except AttributeError:
        return

def requestMediaLinks(media):
    """
    from the media information in 1001 website ({"idItem": idItem, "idObject": idObject})
    request the links with the AJAX request to get_medialink.php
    """
    if isinstance(media, str):
        media = media.replace("'", '"')
        media = json.loads(media)
    requestUrlUrl = "https://www.1001tracklists.com/ajax/get_medialink.php?idObject=" + media["idObject"] + "&idItem=" + media["idItem"]
    jsonResult = json.loads(urlopen(requestUrlUrl).read())

    # soundcloud
    if 'success' in jsonResult and jsonResult["success"]:
        return jsonResult["data"]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    scrapOverview(rangeToParse=range(1, 1000))
    scrapOverview(rangeToParse=range(200, 1000))
