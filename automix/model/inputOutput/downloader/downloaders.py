"""
contains all the functions to download fromn streaming services
"""

from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

from subprocess import PIPE, Popen
import youtube_dl


def oneThousandOneMedia(mediaJson, folder):
    """
    Parse the JSON returned by getMedia function of 1001tracklist
    """
    if "player" in mediaJson and "soundcloud" in mediaJson["player"]:
        return downloadSoundCloud(BeautifulSoup(mediaJson["player"], 'html.parser').find("iframe")["src"], folder)
    elif "player" in mediaJson and "youtube" in mediaJson["player"]:
        print("YOUTUBE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return downloadYoutube(url=BeautifulSoup(mediaJson["player"], 'html.parser').find("iframe")["src"], folder=folder)


def downloadSoundCloud(url, folder):
    """
    Download the soundcloud music from the url
    either https://w.soundcloud.com/player/?url=https://api.soundcloud.com/tracks/419898048
        &show_artwork=true&color=D6DCFE
    or https://api.soundcloud.com/tracks/419898048
    in the specified folder

    returns the filename
    """
    # parse the url to see if it contains a query
    urlParsed = urlparse(url)
    queryParsed = parse_qs(urlParsed.query)
    if 'url' in queryParsed and queryParsed['url']:
        url = queryParsed['url'][0]

    # do the call
    # TODO: enhance the filename parsing. it's not working correctly
    args = ['scdl', '-l', url, '--path', folder, '-c', '--hide-progress']  # '--addtimestamp'
    process = Popen(args, stderr=PIPE)
    output = process.stderr.read()
    splitedOutput = output.decode("utf-8").split("\n")
    if len(splitedOutput) == 7:  # just downloaded
        path = splitedOutput[-3][5:-12]
    else:  # already downloaded
        path = splitedOutput[-2][12:-25] + ".mp3"

    return path


def downloadYoutube(youTubeID="", url="", folder="", path=""):
    """
    Download youtube video fron an ID
    """
    url = "https://www.youtube.com/watch?v=" + youTubeID if youTubeID else url
    outtmpl = folder + '%(title)s.%(ext)s' if folder else path + ".%(ext)s"

    # youTube = YouTube(url)
    # thisStream = youTube.streams.filter(
    #     only_audio=True).order_by('resolution').desc().first()
    # path = folder + youTube.title + ".mp4"

    # if thisStream:
    #     try:  #retrieve the cache
    #         with open(path, 'r'):
    #             pass
    #     except IOError:
    #         thisStream.download(folder)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': outtmpl,
        'gettitle': 1
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        infos = ydl.extract_info(url)
    return str(infos[u"title"]) + ".mp3"  # Track(name=yt.title, path=path)

# print(downloadYoutube(youTubeID="vPIaMZSmLc4", folder="./"))