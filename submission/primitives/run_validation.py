#!/usr/bin/env python3

import argparse
import collections
import contextlib
import glob
import hashlib
import io
import json
import os.path
import pprint
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from distutils import version
from urllib import parse as url_parse

import deep_dircmp
import deepdiff
import docker
import frozendict
import pycurl
import yaml

# To have both stderr and stdout interleaved together.
sys.stderr = sys.stdout

PRIMITIVE_ANNOTATION_REGEX = re.compile(r'^(?P<interface_version>v[^/]+)/(?P<performer_team>[^/]+)/(?P<python_path>[^/]+)/(?P<version>[^/]+)/primitive\.json$')
PIPELINE_REGEX = re.compile(r'^(?P<interface_version>v[^/]+)/(?P<performer_team>[^/]+)/(?P<python_path>[^/]+)/(?P<version>[^/]+)/pipelines/[^/.]+(\.yml|\.json|\.meta)$')
FIX_EGG_VALUE_REGEX = re.compile(r'Fix your #egg=\S+ fragments')
MAIN_REPOSITORY = 'https://gitlab.com/datadrivendiscovery/primitives.git'
MAIN_PROJECT = 'https://gitlab.com/datadrivendiscovery/primitives'

FIXED_PACKAGE_VERSIONS = {
    'v2018.4.18': {
        'tensorflow': '1.8.0',
        'Keras': '2.1.6',
        'torch': '0.3.1',
        'Theano': '1.0.1',
    },
    'v2018.6.5': {
        'tensorflow': '1.8.0',
        'Keras': '2.1.6',
        'torch': '0.3.1',
        'Theano': '1.0.1',
    },
    'v2018.7.10': {
        'tensorflow': '1.8.0',
        'Keras': '2.1.6',
        'torch': '0.3.1',
        'Theano': '1.0.1',
    },
    'v2019.1.21': {
        'tensorflow': '1.12.0',
        'Keras': '2.2.4',
        'torch': '1.0.0',
        'Theano': '1.0.4',
    },
    'v2019.2.12': {
        'tensorflow': '1.12.0',
        'Keras': '2.2.4',
        'torch': '1.0.0',
        'Theano': '1.0.4',
    },
    'v2019.2.18': {
        'tensorflow': '1.12.0',
        'Keras': '2.2.4',
        'torch': '1.0.0',
        'Theano': '1.0.4',
    },
    'v2019.4.4': {
        'tensorflow': '',
        'tensorflow-gpu': '1.12.0',
        'Keras': '2.2.4',
        'torch': '1.0.0',
        'Theano': '1.0.4',
    },
    'v2019.5.8': {
        'tensorflow': '',
        'tensorflow-gpu': '1.12.2',
        'Keras': '2.2.4',
        'torch': '1.0.1.post2',
        'Theano': '1.0.4',
        # Temporary for this version. In the future this will be a core dependency.
        'scipy': '1.2.1',
    },
}

parser = argparse.ArgumentParser(description="Run primitive annotation validation.")
parser.add_argument(
    '-d', '--devel', action='store_true',
    help="validate annotations from latest stable version against devel version of core packages"
)
parser.add_argument(
    '-u', '--username',
    help="username to use for accessing gitlab.datadrivendiscovery.org"
)
parser.add_argument(
    '-p', '--password',
    help="password to use for accessing gitlab.datadrivendiscovery.org"
)
parser.add_argument(
    '-c', '--clean', action='store_true',
    help="remove existing Docker images before downloading new ones"
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '-a', '--all', action='store_true',
    help="validate all primitive annotations and not just newly added"
)
group.add_argument('files', metavar='FILENAME', nargs='*', help="primitive annotation to validate, default is all newly added", default=())
arguments = parser.parse_args()

if (arguments.username and not arguments.password) or (not arguments.username and arguments.password):
    parser.error("both --username and --password are required, or none")

docker_client = docker.from_env()


class ValidationError(ValueError):
    pass


def make_hashable(obj):
    if isinstance(obj, dict):
        return frozendict.frozendict({make_hashable(key): make_hashable(value) for key, value in obj.items()})
    elif isinstance(obj, list):
        return tuple(make_hashable(value) for value in obj)
    else:
        return obj


def docker_exec(docker_container, args, *, print_output=True, run_as_nobody=False):
    if run_as_nobody:
        user = 'nobody:nogroup'
    else:
        user = ''

    response = docker_container.client.api.exec_create(docker_container.id, args, user=user)

    captured_output = io.BytesIO()
    for chunk in docker_container.client.api.exec_start(response['Id'], stream=True):
        if print_output:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        captured_output.write(chunk)

    exit_code = docker_container.client.api.exec_inspect(response['Id'])['ExitCode']

    if exit_code != 0:
        # If output was not printed, print it now so that error can be diagnosed.
        if not print_output:
            sys.stdout.buffer.write(captured_output.getvalue())
            sys.stdout.flush()
        raise ValidationError("Docker command exited with non-zero exit code ({exit_code}): {args}".format(exit_code=exit_code, args=args))

    return captured_output.getvalue().decode('utf8')


def store_d3m_lib(docker_container, output_directory):
    python_libs_path = docker_exec(docker_container, ['python3', '-c', 'import site; import sys; sys.stdout.write(site.getsitepackages()[0])'], print_output=False)

    python_libs_path_base = os.path.basename(python_libs_path)

    with tempfile.TemporaryFile() as python_lib_tar_file:
        for chunk in docker_container.get_archive('{python_libs_path}/d3m'.format(python_libs_path=python_libs_path))[0].stream():
            python_lib_tar_file.write(chunk)

        python_lib_tar_file.seek(0)
        python_lib_tar = tarfile.open(fileobj=python_lib_tar_file)

        for member in python_lib_tar.getmembers():
            # We care only about regular files.
            if member.type not in [tarfile.REGTYPE, tarfile.AREGTYPE]:
                continue

            # Sanity checking.
            if not member.name.startswith(python_libs_path_base):
                continue
            if '..' in member.name:
                continue

            file_name = os.path.join(output_directory, member.name)

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'xb') as file:
                shutil.copyfileobj(python_lib_tar.extractfile(member), file)


def get_annotation_generation_stderr(docker_container):
    stderr_tar_file = io.BytesIO()
    for chunk in docker_container.get_archive('/tmp/stderr')[0].stream():
        stderr_tar_file.write(chunk)

    stderr_tar_file.seek(0)
    stderr_tar = tarfile.open(fileobj=stderr_tar_file)

    return stderr_tar.extractfile('stderr').read().decode('utf8')


def private_pip_uri_access(uri):
    """
    Convert a pip installation URI to something which can be accessed.
    """

    # "git+git@git.myproject.org:MyProject" format cannot be parsed with urlparse.
    if uri.startswith('git+git@'):
        return uri

    parsed_uri = url_parse.urlparse(uri)

    if parsed_uri.hostname != 'gitlab.datadrivendiscovery.org':
        # Not a private URI.
        return uri

    # Not a good practice, but this is primitive author's problem. They could just make a repository public.
    if parsed_uri.username and parsed_uri.password:
        # Private URI already has username and password.
        return uri

    # Remove anything before "@", if it exists.
    _, _, netloc = parsed_uri.netloc.rpartition('@')

    if 'CI_JOB_TOKEN' in os.environ:
        # If we are running on GitLab CI.
        parsed_uri = parsed_uri._replace(netloc='gitlab-ci-token:{CI_JOB_TOKEN}@{netloc}'.format(
            CI_JOB_TOKEN=os.environ['CI_JOB_TOKEN'],
            netloc=netloc,
        ))
    elif arguments.username and arguments.password:
        # If username and password were provided.
        parsed_uri = parsed_uri._replace(netloc='{username}:{password}@{netloc}'.format(
            username=arguments.username,
            password=arguments.password,
            netloc=netloc,
        ))
    else:
        raise ValidationError("A gitlab.datadrivendiscovery.org URI encountered, but no username and password provided.")

    return url_parse.urlunparse(parsed_uri)


def private_git_uri_access(uri):
    """
    Convert a git URI to something which can be accesses.
    """

    parsed_uri = url_parse.urlparse(uri)

    if parsed_uri.hostname != 'gitlab.datadrivendiscovery.org':
        # Not a private URI.
        return uri

    # Remove anything before "@", if it exists.
    _, _, netloc = parsed_uri.netloc.rpartition('@')

    if 'CI_JOB_TOKEN' in os.environ:
        # If we are running on GitLab CI.
        parsed_uri = parsed_uri._replace(netloc='gitlab-ci-token:{CI_JOB_TOKEN}@{netloc}'.format(
            CI_JOB_TOKEN=os.environ['CI_JOB_TOKEN'],
            netloc=netloc,
        ))
        parsed_uri = parsed_uri._replace(scheme='https')
    elif arguments.username and arguments.password:
        # If username and password were provided.
        parsed_uri = parsed_uri._replace(netloc='{username}:{password}@{netloc}'.format(
            username=arguments.username,
            password=arguments.password,
            netloc=netloc,
        ))
        parsed_uri = parsed_uri._replace(scheme='https')
    else:
        # We are running somewhere else. Assume that SSH git URIs will work.
        parsed_uri = parsed_uri._replace(netloc='git@{netloc}'.format(
            netloc=netloc,
        ))
        parsed_uri = parsed_uri._replace(scheme='ssh')

    return url_parse.urlunparse(parsed_uri)


def filter_for_devel(files):
    known_versions = [version.LooseVersion(directory.name[1:]) for directory in os.scandir('.') if directory.is_dir() and directory.name.startswith('v')]

    max_version = max(known_versions)

    return [file for file in files if file.startswith('v{max_version}'.format(max_version=max_version))]


last_progress_call = None


def curl_progress(download_total, downloaded, upload_total, uploaded):
    global last_progress_call

    # Output at most once every 10 seconds.
    now = time.time()
    if last_progress_call is None or now - last_progress_call > 10:
        last_progress_call = now

        print("Downloaded {downloaded}/{download_total} B".format(
            downloaded=downloaded,
            download_total=download_total,
        ), flush=True)


def validate_file(file_path):
    print(">>> Validating file '{file_changed}'.".format(file_changed=file_path), flush=True)

    match = PIPELINE_REGEX.search(file_path)
    if match:
        file_path = '{interface_version}/{performer_team}/{python_path}/{version}/primitive.json'.format(**match.groupdict())
        print(">>> File is a pipeline. Rewriting to primitive annotation file '{file_path}'.".format(file_path=file_path), flush=True)

    match = PRIMITIVE_ANNOTATION_REGEX.search(file_path)
    if not match:
        raise ValidationError("Invalid file path.")

    # A dict capturing segments of the file path.
    entry = match.groupdict()

    with open(file_path, 'r') as annotation_file:
        annotation = json.load(annotation_file)

    interface_version = 'v{version}'.format(version=annotation['primitive_code']['interfaces_version'])
    if interface_version != entry['interface_version']:
        raise ValidationError("'primitive_code.interfaces_version' metadata entry ('{interface_version_metadata}') does not match 'interface_version' path segment ('{interface_version_segment}').".format(
            interface_version_metadata=interface_version,
            interface_version_segment=entry['interface_version'],
        ))

    source_name = annotation['source'].get('name', None)
    if source_name != entry['performer_team']:
        raise ValidationError("'source.name' metadata entry ('{source_name}') does not match 'performer_team' path segment ('{performer_team}').".format(
            source_name=source_name,
            performer_team=entry['performer_team'],
        ))

    if annotation['python_path'] != entry['python_path']:
        raise ValidationError("'python_path' metadata entry ('{python_path_metadata}') does not match 'python_path' path segment ('{python_path_segment}').".format(
            python_path_metadata=annotation['python_path'],
            python_path_segment=entry['python_path'],
        ))

    if annotation['version'] != entry['version']:
        raise ValidationError("'version' metadata entry ('{version_metadata}') does not match 'version' path segment ('{version_segment}').".format(
            version_metadata=annotation['version'],
            version_segment=entry['version'],
        ))

    for globbed_file_path in glob.glob('{interface_version}/*/{python_path}/*/primitive.json'.format(interface_version=interface_version, python_path=annotation['python_path'])):
        if globbed_file_path == file_path:
            continue

        raise ValidationError("Only one primitive version can exist for a primitive's Python path and interface version. There is also '{globbed_file_path}'.".format(
            globbed_file_path=globbed_file_path,
        ))

    return interface_version, annotation


def validate_installation(primitive_names, interface_version, installation, annotations):
    global error_count

    print(">>> Validating installation {primitive_names}.".format(primitive_names=primitive_names), flush=True)

    if arguments.devel:
        image_ubuntu_version = 'bionic-python36'
        image_interface_version = 'devel'
        docker_image = f'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-{image_ubuntu_version}-{image_interface_version}'
        new_cli_interface = True
    elif interface_version in ['v2019.4.4']:
        image_ubuntu_version = 'bionic-python36'
        image_interface_version = interface_version
        docker_image = f'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-{image_ubuntu_version}-{image_interface_version}'
        new_cli_interface = False
    else:
        image_ubuntu_version = 'bionic-python36'
        image_interface_version = interface_version
        docker_image = f'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-{image_ubuntu_version}-{image_interface_version}'
        new_cli_interface = True

    # Run a container which sleeps until we stop it. First we remove all existing Docker images to make the space for the new one.
    if arguments.clean:
        print(">>> Removing exiting Docker images.", flush=True)
        for image in docker_client.images.list():
            # If we already have the image we want to run, we do not remove it.
            if docker_image in image.tags:
                continue

            # Otherwise we remove it.
            print(">>> Removing Docker image '{docker_image}'.".format(docker_image=image.tags[0] if image.tags else image.id), flush=True)
            docker_client.images.remove(image.id, force=True)

    # Then we pull the latest version of the image.
    print(">>> Pulling Docker image '{docker_image}'.".format(docker_image=docker_image), flush=True)
    previous_chunk = None
    for chunk in docker_client.api.pull(docker_image, stream=True, decode=True):
        if 'status' in chunk:
            if previous_chunk == chunk['status']:
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                print(chunk['status'], flush=True)

            previous_chunk = chunk['status']

    print(">>> Running Docker image '{docker_image}'.".format(docker_image=docker_image), flush=True)
    docker_container = docker_client.containers.run(
        docker_image,
        ['sleep', 'infinity'],
        detach=True, auto_remove=True, stop_signal='SIGKILL',
    )

    pip_version = docker_exec(docker_container, ['python3', '-c', 'import pip; print(pip.__version__)'], print_output=False)
    can_use_process_dependency_links = version.LooseVersion(pip_version) < version.LooseVersion('19.0.0')

    existing_python_lib_dir = None
    updated_python_lib_dir = None
    already_updated_apt = False
    try:
        print(">>> Existing Python packages installed:", flush=True)
        existing_packages = set(docker_exec(docker_container, ['pip3', 'freeze', '--disable-pip-version-check']).splitlines())

        # We copy and store d3m package. It had happened that primitives modified its files directly by accident.
        print(">>> Copying d3m Python package.", flush=True)
        existing_python_lib_dir = tempfile.TemporaryDirectory()
        store_d3m_lib(docker_container, existing_python_lib_dir.name)

        print(">>> Installing primitive dependencies.", flush=True)

        if installation == ():
            print(">>> ERROR {primitive_names}: 'installation' metadata is empty. Primitive canot be installed.".format(primitive_names=primitive_names), flush=True)
            error_count += len(primitive_names)
            return

        for installation_entry in installation:
            if installation_entry['type'] == 'PIP' and 'package' in installation_entry:
                registry = installation_entry.get('registry', 'https://pypi.python.org/simple')
                print(">>> Installing Python package '{package}=={version}' from '{registry}'.".format(package=installation_entry['package'], version=installation_entry['version'], registry=registry), flush=True)
                args = ['pip3', '--disable-pip-version-check', 'install', '--process-dependency-links', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--exists-action', 'w', '--index-url', registry, '{package}=={version}'.format(package=installation_entry['package'], version=installation_entry['version'])]
                if not can_use_process_dependency_links:
                    args.remove('--process-dependency-links')
                docker_exec(docker_container, args)

            elif installation_entry['type'] == 'PIP' and 'package_uri' in installation_entry:
                print(">>> Installing Python package '{package_uri}'.".format(package_uri=installation_entry['package_uri']), flush=True)
                # git+git scheme is not supported, and other URIs can be parsed with urlparse.
                parsed_uri = url_parse.urlparse(installation_entry['package_uri'])

                # Not a git URI. Just directly install, without "--editable" argument.
                if not parsed_uri.scheme.startswith('git'):
                    package_uri = url_parse.urlunparse(parsed_uri)

                    # Add username and password for private URIs.
                    package_uri = private_pip_uri_access(package_uri)

                    args = ['pip3', '--disable-pip-version-check', 'install', '--process-dependency-links', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--exists-action', 'w', package_uri]
                    if not can_use_process_dependency_links:
                        args.remove('--process-dependency-links')
                    docker_exec(docker_container, args)

                else:
                    # In these versions we allowed "egg" value to be missing, so we create it ourselves.
                    if image_interface_version in ['v2017.12.27', 'v2018.1.5']:
                        # To install with "--editable" argument an "egg" fragment argument has to be provided.
                        # We parse it here and set it to a random UUID to assure it is unique.
                        if parsed_uri.fragment:
                            parsed_fragment = url_parse.parse_qs(parsed_uri.fragment, strict_parsing=True)
                        else:
                            parsed_fragment = {}

                        if 'egg' not in parsed_fragment:
                            parsed_fragment['egg'] = [uuid.uuid4()]

                        parsed_uri = parsed_uri._replace(fragment=url_parse.urlencode(parsed_fragment, doseq=True, safe='/', quote_via=url_parse.quote))

                    package_uri = url_parse.urlunparse(parsed_uri)

                    # Add username and password for private URIs.
                    package_uri = private_pip_uri_access(package_uri)

                    # We install with "--editable" so that packages can have access to their git repositories.
                    # For example, they might need it to compute installation git commit hash for their metadata.
                    args = ['pip3', '--disable-pip-version-check', 'install', '--process-dependency-links', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--exists-action', 'w', '--editable', package_uri]
                    if not can_use_process_dependency_links:
                        args.remove('--process-dependency-links')
                    installation_log = docker_exec(docker_container, args)

                    # In newer versions we require "egg" value to match the package name.
                    if image_interface_version not in ['v2017.12.27', 'v2018.1.5'] and FIX_EGG_VALUE_REGEX.search(installation_log):
                        print(">>> ERROR {primitive_names}: Package URI's 'egg' value does not match package name installed.".format(primitive_names=primitive_names), flush=True)
                        error_count += len(primitive_names)
                        return

            elif installation_entry['type'] == 'DOCKER':
                print(">>> Validating Docker image '{image_name}@{image_digest}'.".format(image_name=installation_entry['image_name'], image_digest=installation_entry['image_digest']), flush=True)
                previous_chunk = None
                for chunk in docker_client.api.pull('{image_name}@{image_digest}'.format(image_name=installation_entry['image_name'], image_digest=installation_entry['image_digest']), stream=True, decode=True):
                    if 'status' in chunk:
                        if previous_chunk == chunk['status']:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        else:
                            print(chunk['status'], flush=True)

                        previous_chunk = chunk['status']

            elif installation_entry['type'] == 'UBUNTU':
                print(">>> Installing Ubuntu package '{package}'.".format(package=installation_entry['package']), flush=True)
                if not already_updated_apt:
                    docker_exec(docker_container, ['apt-get', 'update', '-q', '-q'])
                    already_updated_apt = True
                docker_exec(docker_container, ['apt-get', 'install', '--yes', '--force-yes', '--no-install-recommends', installation_entry['package']])

            elif installation_entry['type'] in ['FILE', 'TGZ']:
                print(">>> Downloading and computing digest for file '{file_uri}'.".format(file_uri=installation_entry['file_uri']), flush=True)

                hash = hashlib.sha256()
                downloaded = 0
                start = time.time()

                def write(data):
                    nonlocal hash
                    nonlocal downloaded

                    hash.update(data)
                    downloaded += len(data)

                while True:
                    try:
                        with contextlib.closing(pycurl.Curl()) as curl:
                            curl.setopt(curl.URL, installation_entry['file_uri'])
                            curl.setopt(curl.WRITEFUNCTION, write)
                            curl.setopt(curl.NOPROGRESS, False)
                            curl.setopt(curl.FOLLOWLOCATION, True)
                            curl.setopt(curl.XFERINFOFUNCTION, curl_progress)
                            curl.setopt(curl.LOW_SPEED_LIMIT, 30 * 1024)
                            curl.setopt(curl.LOW_SPEED_TIME, 30)
                            curl.setopt(curl.RESUME_FROM, downloaded)

                            curl.perform()
                            break

                    except pycurl.error as error:
                        if error.args[0] == pycurl.E_OPERATION_TIMEDOUT:
                            # If timeout, retry/resume.
                            print(">>> Timeout. Retrying.", flush=True)
                        else:
                            raise

                end = time.time()

                print(">>> Downloaded {downloaded} B in {seconds} second(s).".format(
                    downloaded=downloaded,
                    seconds=end - start,
                ), flush=True)

                if installation_entry['file_digest'] != hash.hexdigest():
                    print(">>> ERROR {primitive_names}: Digest for downloaded file does not match one from metadata. Metadata digest: {metadata_digest}. Computed digest: {computed_digest}.".format(
                        primitive_names=primitive_names,
                        metadata_digest=installation_entry['file_digest'],
                        computed_digest=hash.hexdigest(),
                    ), flush=True)
                    error_count += len(primitive_names)
                    return

            else:
                raise ValidationError("Unknown dependency type: {type}".format(type=installation_entry['type']))

        print(">>> All Python packages installed now:", flush=True)
        updated_packages = set(docker_exec(docker_container, ['pip3', 'freeze', '--disable-pip-version-check']).splitlines())

        new_packages = updated_packages - existing_packages
        if len(new_packages):
            print(">>> New (or updated) Python packages:", flush=True)
            for new_package in new_packages:
                print(new_package, flush=True)

        expected_packages = existing_packages - updated_packages
        if len(expected_packages):
            print(">>> WARNING {primitive_names}: Existing Python packages had their versions changed. Please review these changes.".format(primitive_names=primitive_names), flush=True)
            print(">>> Existing Python packages with versions changed:", flush=True)
            for expected_package in expected_packages:
                print(expected_package, flush=True)

        installed_packages_json = docker_exec(docker_container, ['pip3', 'list', '--disable-pip-version-check', '--format', 'json'], print_output=False)
        try:
            installed_packages = json.loads(installed_packages_json)
        except json.decoder.JSONDecodeError:
            print(">>> Generated Python packages list:", flush=True)
            sys.stdout.write(installed_packages_json)
            sys.stdout.flush()
            raise

        mismatched_versions = {}
        for package in installed_packages:
            expected_version = FIXED_PACKAGE_VERSIONS.get(image_interface_version, {}).get(package['name'], None)
            if expected_version is not None and expected_version != package['version']:
                mismatched_versions[package['name']] = (expected_version, package['version'])

        if len(mismatched_versions):
            print(">>> ERROR {primitive_names}: Python packages with fixed version installed with a different version.".format(primitive_names=primitive_names), flush=True)
            print(">>> Python packages with versions mismatch:", flush=True)
            for package_name, versions in mismatched_versions.items():
                print(package_name, 'expected: {expected}'.format(expected=versions[0]), 'installed: {installed}'.format(installed=versions[1]), flush=True)
            error_count += len(primitive_names)
            return

        # We copy and store d3m package again, to compare.
        print(">>> Copying d3m Python package.", flush=True)
        updated_python_lib_dir = tempfile.TemporaryDirectory()
        store_d3m_lib(docker_container, updated_python_lib_dir.name)

        print(">>> Comparing d3m Python package.", flush=True)
        dir_comparison = deep_dircmp.DeepDirCmp(existing_python_lib_dir.name, updated_python_lib_dir.name, hide=[], ignore=[])

        left_only = dir_comparison.get_left_only_recursive()
        if len(left_only):
            print(">>> ERROR {primitive_names}: d3m Python package had its files removed. This is not allowed.".format(primitive_names=primitive_names), flush=True)
            print(">>> Files removed:", flush=True)
            for file in left_only:
                print(file, flush=True)
            error_count += len(primitive_names)
            return

        right_only = dir_comparison.get_right_only_recursive()
        if len(right_only):
            print(">>> ERROR {primitive_names}: d3m Python package had its files added. This is not allowed.".format(primitive_names=primitive_names), flush=True)
            print(">>> Files added:", flush=True)
            for file in right_only:
                print(file, flush=True)
            error_count += len(primitive_names)
            return

        common_funny = dir_comparison.get_common_funny_recursive()
        if len(common_funny):
            print(">>> ERROR {primitive_names}: d3m Python package had its files changed in a funny way. This is not allowed.".format(primitive_names=primitive_names), flush=True)
            print(">>> Files changed in a funny way:", flush=True)
            for file in common_funny:
                print(file, flush=True)
            error_count += len(primitive_names)
            return

        diff_files = dir_comparison.get_diff_files_recursive()

        if len(diff_files):
            print(">>> ERROR {primitive_names}: d3m Python packages had its files changed. This is not allowed.".format(primitive_names=primitive_names), flush=True)
            print(">>> Files changed:", flush=True)
            for file in diff_files:
                print(file, flush=True)
            error_count += len(primitive_names)
            return

        print(">>> Checking Python packages.", flush=True)
        try:
            docker_exec(docker_container, ['pip3', '--disable-pip-version-check', 'check'])
        except ValidationError:
            print(">>> ERROR {primitive_names}: Checking Python packages failed which probably means that existing Python packages had their versions changed to incompatible versions. This is not allowed. Consider changing dependencies in your primitive to match existing Python packages' versions. Or contact existing primitves' authors to determine common versions for shared dependencies. If this does not work out, bring it to the working group to discuss fixing a version of conflicting dependencies for all primitives.".format(primitive_names=primitive_names), flush=True)
            error_count += len(primitive_names)
            return

        print(">>> Validating annotations {primitive_names}.".format(primitive_names=primitive_names), flush=True)

        for annotation in annotations:
            python_path = annotation['python_path']
            primitive_name = '{interface_version}/{python_path}/{version}'.format(interface_version=interface_version, python_path=python_path, version=annotation['version'])

            try:
                # We run this as non-root to make sure primitive can run without a root user.
                print(">>> Generating JSON-serialized metadata for '{primitive_name}'.".format(primitive_name=primitive_name), flush=True)
                try:
                    if new_cli_interface:
                        generated_json_annotation = docker_exec(docker_container, ['bash', '-c', 'python3 -m d3m index describe -i 4 {python_path} 2> /tmp/stderr'.format(python_path=python_path)], print_output=False, run_as_nobody=True)
                    else:
                        generated_json_annotation = docker_exec(docker_container, ['bash', '-c', 'python3 -m d3m.index describe -i 4 {python_path} 2> /tmp/stderr'.format(python_path=python_path)], print_output=False, run_as_nobody=True)
                except ValidationError:
                    stderr = get_annotation_generation_stderr(docker_container)
                    print(">>> Stderr during generation:", flush=True)
                    sys.stdout.write(stderr)
                    sys.stdout.flush()
                    raise

                if arguments.devel:
                    print(">>> SUCCESS '{primitive_name}': Metadata generated.".format(primitive_name=primitive_name), flush=True)
                else:
                    try:
                        generated_annotation = json.loads(generated_json_annotation)
                    except json.decoder.JSONDecodeError:
                        print(">>> Generated JSON-serialized metadata:", flush=True)
                        sys.stdout.write(generated_json_annotation)
                        sys.stdout.flush()
                        raise

                    print(">>> Comparing generated metadata.", flush=True)
                    diff = deepdiff.DeepDiff(generated_annotation, annotation, verbose_level=2)

                    if diff == {}:
                        print(">>> SUCCESS '{primitive_name}': Metadata matches.".format(primitive_name=primitive_name), flush=True)
                    else:
                        print(">>> ERROR '{primitive_name}': Metadata does not match.".format(primitive_name=primitive_name), flush=True)
                        pprint.pprint(diff)
                        sys.stdout.flush()
                        error_count += 1
                        continue

                pipeline_paths = glob.glob('{interface_version}/{performer_team}/{python_path}/{version}/pipelines/*.yml'.format(
                    interface_version=interface_version, performer_team=annotation['source']['name'],
                    python_path=annotation['python_path'], version=annotation['version'],
                ))
                pipeline_paths += glob.iglob('{interface_version}/{performer_team}/{python_path}/{version}/pipelines/*.json'.format(
                    interface_version=interface_version, performer_team=annotation['source']['name'],
                    python_path=annotation['python_path'], version=annotation['version'],
                ))

                if len(pipeline_paths):
                    print(">>> Validating pipelines {pipeline_paths}.".format(pipeline_paths=pipeline_paths), flush=True)

                    pipeline_error = False
                    for pipeline_path in pipeline_paths:
                        with open(pipeline_path, 'r') as pipeline_file:
                            if pipeline_path.endswith('.yml'):
                                pipeline = yaml.safe_load(pipeline_file)
                            elif pipeline_path.endswith('.json'):
                                pipeline = json.load(pipeline_file)
                            else:
                                assert False, pipeline_path

                        pipeline_id = os.path.splitext(os.path.basename(pipeline_path))[0]

                        if pipeline_id != pipeline['id']:
                            print(">>> ERROR '{primitive_name}': Pipeline '{pipeline_path}' filename does not match its ID.".format(
                                primitive_name=primitive_name, pipeline_path=pipeline_path,
                            ), flush=True)
                            error_count += 1
                            pipeline_error = True
                            break

                    if pipeline_error:
                        continue

                    pipeline_names = []
                    # These pipelines are tested as standard pipelines, others not.
                    pipelines_with_meta = set()

                    bytes = io.BytesIO()
                    with tarfile.open(mode='w', fileobj=bytes) as tar:
                        for pipeline_path in pipeline_paths:
                            pipeline_name = os.path.basename(pipeline_path)
                            tar.add(pipeline_path, pipeline_name)
                            pipeline_names.append(pipeline_name)

                            meta_path = os.path.splitext(pipeline_path)[0] + '.meta'
                            if os.path.exists(meta_path):
                                with open(meta_path, 'r') as meta_file:
                                    # For now, we just check that it is a valid JSON.
                                    # TODO: Migrate to using pipeline run.
                                    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/202
                                    json.load(meta_file)

                                meta_name = os.path.splitext(pipeline_name)[0] + '.meta'
                                tar.add(meta_path, meta_name)
                                pipelines_with_meta.add(pipeline_name)

                    if pipeline_error:
                        continue

                    bytes.seek(0)
                    docker_exec(docker_container, ['rm', '-rf', '/tmp/pipelines'])
                    docker_exec(docker_container, ['mkdir', '/tmp/pipelines'])
                    docker_container.put_archive('/tmp/pipelines', bytes)

                    for pipeline_name in pipeline_names:
                        if pipeline_name in pipelines_with_meta:
                            # Checked as a standard pipeline (Dataset inputs and DataFrame predictions output).
                            if new_cli_interface:
                                docker_exec(docker_container, [
                                    'python3',
                                    '-m',
                                    'd3m',
                                    '--pipelines-path', '/tmp/pipelines',
                                    'pipeline',
                                    'describe',
                                    '/tmp/pipelines/{pipeline_name}'.format(pipeline_name=pipeline_name),
                                ], print_output=False, run_as_nobody=True)
                            else:
                                docker_exec(docker_container, [
                                    'python3',
                                    '-m',
                                    'd3m.metadata.pipeline',
                                    '--check',
                                    '--pipelines-path', '/tmp/pipelines',
                                    '/tmp/pipelines/{pipeline_name}'.format(pipeline_name=pipeline_name),
                                ], print_output=False, run_as_nobody=True)
                        else:
                            # Checked as a non-standard pipeline (any inputs and outputs are allowed).
                            if new_cli_interface:
                                docker_exec(docker_container, [
                                    'python3',
                                    '-m',
                                    'd3m',
                                    '--pipelines-path', '/tmp/pipelines',
                                    'pipeline',
                                    'describe',
                                    '--not-standard-pipeline',
                                    '/tmp/pipelines/{pipeline_name}'.format(pipeline_name=pipeline_name),
                                ], print_output=False, run_as_nobody=True)
                            else:
                                docker_exec(docker_container, [
                                    'python3',
                                    '-m',
                                    'd3m.metadata.pipeline',
                                    '--check',
                                    '--not-standard-pipeline',
                                    '--pipelines-path', '/tmp/pipelines',
                                    '/tmp/pipelines/{pipeline_name}'.format(pipeline_name=pipeline_name),
                                ], print_output=False, run_as_nobody=True)

            except Exception as error:
                error_count += 1
                print(">>> ERROR '{primitive_name}': {error}".format(primitive_name=primitive_name, error=error), flush=True)

                if not isinstance(error, ValidationError):
                    traceback.print_exc()
                    sys.stdout.flush()

    finally:
        print(">>> Stopping Docker container for image '{docker_image}'.".format(docker_image=docker_image), flush=True)
        docker_container.stop()

        if existing_python_lib_dir is not None:
            existing_python_lib_dir.cleanup()
        if updated_python_lib_dir is not None:
            updated_python_lib_dir.cleanup()


main_repository = private_git_uri_access(MAIN_REPOSITORY)
master_branch_main_project = os.environ.get('GITLAB_CI', False) and os.environ['CI_COMMIT_REF_NAME'] == 'master' and os.environ['CI_PROJECT_URL'] == MAIN_PROJECT

if arguments.devel:
    print(">>> Validating against devel version.", flush=True)

# If argument is set, we validate all annotations in the repo.
# We also validate all annotations when running on the master branch of the main project.
if arguments.all or master_branch_main_project:
    print(">>> Validating all annotations.", flush=True)

    prefix = os.path.join('.', 'v')
    # With "relpath" we remove "./" prefix.
    files_changed = [os.path.relpath(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk('.') for filename in filenames if dirpath.startswith(prefix) and filename != '.gitignore']

    if arguments.devel:
        files_changed = filter_for_devel(files_changed)

# If a list of files to validate was provided.
elif arguments.files:
    print(">>> Validating listed annotations.", flush=True)

    # Map to relative paths to increase chance that the file path matches the regex.
    files_changed = [os.path.relpath(file_path) for file_path in arguments.files]

# Otherwise we find only the files changed from the master branch of the main repository.
else:
    print(">>> Validating newly added annotations.", flush=True)

    # We allow it to fail.
    subprocess.run([
        'git', 'remote', 'remove', 'upstream',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, encoding='utf8')

    subprocess.run([
        'git', 'remote', 'add', '-f', 'upstream', main_repository,
    ], stdout=sys.stdout, stderr=sys.stderr, check=True, encoding='utf8')

    subprocess.run([
        'git', 'remote', 'update',
    ], stdout=sys.stdout, stderr=sys.stderr, check=True, encoding='utf8')

    # A list of files which changed between the master branch in the main repository and current branch/repository
    # running this validator. This is less strict than GitLab "changes" view and really shows just the diff in the
    # content and does not take into the account different git histories. It is ".." diff and not GitLab's "..." diff.
    # We do not care about renaming files and want to see them as removing and adding.
    changes = subprocess.run([
        'git', 'diff', '--name-status', '--no-renames', 'remotes/upstream/master', 'HEAD',
    ], stdout=subprocess.PIPE, stderr=sys.stderr, check=True, encoding='utf8').stdout.splitlines()

    files_changed = []
    for change in changes:
        change_type, _, file_changed = change.partition('\t')

        # We skip deletions of primitive annotations. But only those deletions.
        if change_type == 'D' and file_changed.startswith('v'):
            continue

        files_changed.append(file_changed)

    if arguments.devel:
        files_changed = filter_for_devel(files_changed)


error_count = 0
installations = collections.defaultdict(list)

# Now we validate each changed file.
for file_path in files_changed:
    try:
        interface_version, annotation = validate_file(file_path)
        installation = make_hashable(annotation.get('installation', ()))
        if annotation not in installations[(interface_version, installation)]:
            installations[(interface_version, installation)].append(annotation)
    except Exception as error:
        error_count += 1
        print(">>> ERROR '{file_path}': {error}".format(file_path=file_path, error=error), flush=True)

        if not isinstance(error, ValidationError):
            traceback.print_exc()
            sys.stdout.flush()


for (interface_version, installation), annotations in installations.items():
    # We validate in "validate_file" that "python_path" and "version" exist.
    primitive_names = ['{interface_version}/{python_path}/{version}'.format(interface_version=interface_version, python_path=annotation['python_path'], version=annotation['version']) for annotation in annotations]

    try:
        validate_installation(primitive_names, interface_version, installation, annotations)
    except Exception as error:
        error_count += len(primitive_names)
        print(">>> ERROR {primitive_names}: {error}".format(primitive_names=primitive_names, error=error), flush=True)

        if not isinstance(error, ValidationError):
            traceback.print_exc()
            sys.stdout.flush()


if error_count:
    print(">>> There were {error_count} error(s).".format(error_count=error_count), flush=True)
else:
    print(">>> There were no errors.", flush=True)
sys.exit(bool(error_count))
