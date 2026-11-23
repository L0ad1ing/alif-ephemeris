"""Tiny HEASARC TAP client returning rows from a VOTable (development tool)."""
import socket, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

BASE = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"


def query(adql, timeout=180):
    socket.setdefaulttimeout(timeout)
    url = BASE + "?" + urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": adql})
    root = ET.fromstring(urllib.request.urlopen(url).read())
    ns = {"v": root.tag.split("}")[0].strip("{")}
    info = root.find(".//v:INFO[@name='QUERY_STATUS']", ns)
    if info is not None and info.get("value") == "ERROR":
        raise RuntimeError(info.text)
    names = [f.get("name") for f in root.findall(".//v:FIELD", ns)]
    rows = []
    for tr in root.findall(".//v:TABLEDATA/v:TR", ns):
        rows.append([td.text for td in tr.findall("v:TD", ns)])
    return names, rows
