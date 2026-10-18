#!/usr/bin/env python3
"""Assemble signed quant and migration POMs, optionally including a JAX4J release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET

NS = 'http://maven.apache.org/POM/4.0.0'
ET.register_namespace('', NS)

def tag(name):
    return '{' + NS + '}' + name


def add(parent, name, text=None):
    child = ET.SubElement(parent, tag(name))
    child.text = text
    return child


def collect(tree, module, stage):
    root = tree / module if module else tree
    pom = ET.parse(root / 'pom.xml').getroot()
    artifact = pom.findtext(tag('artifactId'))
    parent = pom.find(tag('parent'))
    version = pom.findtext(tag('version')) or parent.findtext(tag('version'))
    if version.endswith('-SNAPSHOT'):
        raise SystemExit('Snapshot version: ' + artifact)
    destination = stage / 'com/nablatensor' / artifact / version
    destination.mkdir(parents=True, exist_ok=True)
    prefix = artifact + '-' + version
    shutil.copy2(root / 'pom.xml', destination / (prefix + '.pom'))
    shutil.copy2(root / 'target' / (prefix + '.pom.asc'), destination / (prefix + '.pom.asc'))
    if pom.findtext(tag('packaging'), 'jar') != 'pom':
        for suffix in ['.jar', '-sources.jar', '-javadoc.jar']:
            for signature in ['', '.asc']:
                name = prefix + suffix + signature
                shutil.copy2(root / 'target' / name, destination / name)
    return artifact + ':' + version


def legacy_pom(artifact, config, target=None, notice=None):
    p = ET.Element(tag('project'))
    add(p, 'modelVersion', '4.0.0')
    add(p, 'groupId', config['groupId'])
    add(p, 'artifactId', artifact)
    add(p, 'version', config['version'])
    add(p, 'packaging', 'pom')
    add(p, 'name', artifact + ' migration notice')
    message = config['message'] + ' ' + (notice or ('Replacement: com.nablatensor:' + ':'.join(target) + '.'))
    add(p, 'description', message)
    add(p, 'url', 'https://github.com/nablatensor-dev/quant')
    license = add(add(p, 'licenses'), 'license')
    add(license, 'name', 'The Apache License, Version 2.0')
    add(license, 'url', 'https://www.apache.org/licenses/LICENSE-2.0.txt')
    developer = add(add(p, 'developers'), 'developer')
    add(developer, 'id', 'petrpravda')
    add(developer, 'name', 'Petr Pravda')
    scm = add(p, 'scm')
    add(scm, 'connection', 'scm:git:https://github.com/nablatensor-dev/quant.git')
    add(scm, 'developerConnection', 'scm:git:git@github.com:nablatensor-dev/quant.git')
    add(scm, 'url', 'https://github.com/nablatensor-dev/quant')
    add(scm, 'tag', 'v' + config['version'])
    if target:
        relocation = add(add(p, 'distributionManagement'), 'relocation')
        add(relocation, 'groupId', config['groupId'])
        add(relocation, 'artifactId', target[0])
        add(relocation, 'version', target[1])
        add(relocation, 'message', message)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jax4j-tree', type=Path, help='Include JAX4J only when publishing a new JAX4J release')
    parser.add_argument('--quant-tree', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output / 'repository'
    artifacts = []
    if args.jax4j_tree:
        for module in ['', 'jax4j-core', 'jax4j-engine-vulkan', 'jax4j-engine-wgpu',
                       'jax4j-engine-cuda', 'jax4j-engine-rocm', 'jax4j-simd']:
            artifacts.append(collect(args.jax4j_tree, module, stage))
    for module in ['', 'quant-codegen', 'quant', 'quant-risk', 'quant-cva']:
        artifacts.append(collect(args.quant_tree, module, stage))
    config = json.loads((args.quant_tree / 'release/retired-artifacts.json').read_text())
    for artifact in sorted(set(config['relocations']) | set(config['notices'])):
        root = legacy_pom(artifact, config, config['relocations'].get(artifact), config['notices'].get(artifact))
        directory = stage / 'com/nablatensor' / artifact / config['version']
        directory.mkdir(parents=True)
        pom = directory / (artifact + '-' + config['version'] + '.pom')
        ET.indent(root, space='  ')
        ET.ElementTree(root).write(pom, encoding='UTF-8', xml_declaration=True)
        with pom.open('a') as stream:
            stream.write('\n')
        subprocess.run(['gpg', '--batch', '--pinentry-mode', 'error', '--armor', '--detach-sign',
                        '--output', str(pom) + '.asc', str(pom)], check=True)
        artifacts.append(artifact + ':' + config['version'])
    files = sorted(p for p in stage.rglob('*') if p.is_file())
    for path in files:
        data = path.read_bytes()
        if b'/home/petr' in data:
            raise SystemExit('Personal path found in ' + str(path.relative_to(stage)))
        if path.suffix == '.pom':
            root = ET.fromstring(data)
            for version in root.iter(tag('version')):
                if version.text and version.text.endswith('-SNAPSHOT'):
                    raise SystemExit('Snapshot in release POM: ' + str(path))
        if path.suffix != '.asc':
            subprocess.run(['gpg', '--batch', '--verify', str(path) + '.asc', str(path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for algorithm in ['md5', 'sha1', 'sha256', 'sha512']:
                Path(str(path) + '.' + algorithm).write_text(hashlib.new(algorithm, data).hexdigest())
    bundle = args.output / 'central-bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(stage))
    manifest = {'artifacts': artifacts, 'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
                'relocations': len(config['relocations']), 'retirement_notices': len(config['notices'])}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
