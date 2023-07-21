""" Getting licenses for all packages in a conda environment. """

#  3rd party
import pkg_resources
import prettytable


def get_pkg_license(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'


def get_pkg_url(pkg):
    try:
        lines = pkg.get_metadata_lines('METADATA')
    except:
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('Home-page:'):
            return line[10:]
    return '(URL not found)'


def print_packages_and_licenses():
    t = prettytable.PrettyTable(['Package', 'License', 'URL'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        t.add_row((str(pkg), get_pkg_license(pkg), get_pkg_url(pkg)))
        print(f'{pkg}: {get_pkg_license(pkg)} {get_pkg_url(pkg)}')
    print(t)


if __name__ == "__main__":
    print_packages_and_licenses()
