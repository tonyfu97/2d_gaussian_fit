import pkg_resources

def generate_requirements(packages):
    with open('requirements.txt', 'w') as file:
        for package_name in packages:
            try:
                version = pkg_resources.get_distribution(package_name).version
                file.write(f'{package_name}=={version}\n')
                print(f'Added {package_name}=={version}')
            except pkg_resources.DistributionNotFound:
                print(f'Package {package_name} not found. Skipping...')

if __name__ == '__main__':
    packages = ['numpy', 'scipy', 'matplotlib', 'tqdm']
    generate_requirements(packages)
    print("requirements.txt file has been created.")
