class XmlSerialiser(object):
    @staticmethod
    def xmlDeserialize(path):
        """
        get a track from the SegmXML format: http://www.ifs.tuwien.ac.at/mir/audiosegmentation.html
        """
        tree = ET.parse(path)
        root = tree.getroot()
        track = Track()

        track.name = root.find('metadata').find('song').find('title').text
        track.segments = [
            Segment(segment.attrib['label'], start=segment.attrib['start_sec'], end=segment.attrib['end_sec'])
            for segment in root.find('segmentation').iter('segment')
        ]
        return track
