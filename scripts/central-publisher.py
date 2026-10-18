#!/usr/bin/env python3
"""Upload, inspect or publish a Central bundle using the local Maven server token."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET


def token(settings, server_id):
    root = ET.parse(settings).getroot()
    ns = root.tag.partition('}')[0] + '}' if '}' in root.tag else ''
    for server in root.findall(f'{ns}servers/{ns}server'):
        if server.findtext(ns + 'id') != server_id:
            continue
        def value(name):
            text = server.findtext(ns + name, '')
            text = re.sub(r'\$\{env\.([^}]+)\}', lambda m: os.environ.get(m[1], ''), text)
            if not text or text.startswith('{'):
                raise SystemExit('A plain token or configured environment token is required in the Maven server entry.')
            return text
        return base64.b64encode((value('username') + ':' + value('password')).encode()).decode()
    raise SystemExit('Maven publishing server entry not found.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['upload', 'status', 'publish'])
    parser.add_argument('--settings', type=Path, default=Path.home() / '.m2/settings.xml')
    parser.add_argument('--server', default='central')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--deployment')
    parser.add_argument('--name', default='quant migration')
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    headers = {'Authorization': 'Bearer ' + token(args.settings, args.server)}
    body = b''
    if args.action == 'upload':
        if not args.bundle:
            parser.error('--bundle is required for upload')
        # Upload for validation first. Publishing is a separate explicit command.
        query = urllib.parse.urlencode({'name': args.name, 'publishingType': 'USER_MANAGED'})
        path = '/upload?' + query
        boundary = 'central-' + uuid.uuid4().hex
        headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="bundle"; filename="central-bundle.zip"\r\n'
                'Content-Type: application/octet-stream\r\n\r\n').encode()
        body += args.bundle.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    else:
        if not args.deployment:
            parser.error('--deployment is required')
        deployment = str(uuid.UUID(args.deployment))
        path = '/status?id=' + deployment if args.action == 'status' else '/deployment/' + deployment
    request = urllib.request.Request('https://central.sonatype.com/api/v1/publisher' + path,
                                     data=body, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = response.read().decode()
    except urllib.error.HTTPError as error:
        print('Central request failed:', error.code, error.read().decode()[:4000])
        raise SystemExit(1)
    if args.record:
        args.record.write_text(result + '\n')
    print(result or 'Request accepted.')


if __name__ == '__main__':
    main()
