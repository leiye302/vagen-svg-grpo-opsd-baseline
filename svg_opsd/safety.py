"""Same offline collection safety boundary, applied identically to both arms."""
import re
from defusedxml import ElementTree


def svg_safety(svg):
    if not isinstance(svg, str) or not svg.strip():
        return False, 'empty_svg'
    if len(svg.encode('utf-8')) > 1024 * 1024:
        return False, 'svg_too_large'
    if re.search(r'<!DOCTYPE|<!ENTITY|<\?xml-stylesheet', svg, re.I):
        return False, 'external_declaration'
    try:
        root = ElementTree.fromstring(svg)
    except Exception:
        return False, 'xml_invalid'
    if root.tag.split('}')[-1].lower() != 'svg':
        return False, 'root_not_svg'
    count = 0
    for node in root.iter():
        count += 1
        if count > 10000:
            return False, 'too_many_elements'
        tag = node.tag.split('}')[-1].lower()
        if tag in ('script', 'foreignobject', 'iframe', 'audio', 'video', 'image'):
            return False, 'active_or_external_element'
        for key, value in node.attrib.items():
            key = key.split('}')[-1].lower()
            if key.startswith('on'):
                return False, 'event_handler'
            if key in ('href', 'src') and not value.strip().startswith('#'):
                return False, 'external_reference'
            for ref in re.findall(r'url\s*\((.*?)\)', value, re.I):
                if not ref.strip(' \t\r\n\"\'').startswith('#'):
                    return False, 'external_url'
        style = node.text if tag == 'style' else ''
        if style and ('@import' in style.lower() or re.search(r'url\s*\(\s*[^#]', style, re.I)):
            return False, 'external_style'
    return True, 'safe'

