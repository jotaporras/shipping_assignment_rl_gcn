from setuptools import setup, find_packages

print("Running setup")
setup(
    name="shipping-allocation",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    # package_data={
    #   '': ['*.csv', '*.npy'],
    # },
    version="0.0.1",
    install_requires=["gym", "numpy", "pandas", "joblib", "ortools"],
)
