import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "MLAutoRegressor"
USER_NAME = "Rohanbagulwar"

setuptools.setup(
    name=f"{PROJECT_NAME}",
    version="0.0.3",
    author=['Rohan Bagulwar','ketan gangal'],
    author_email=["rohanbagulwarrb@gmail.com","ketangangal98@gmail.com"],
    description="its an implimentation of automatic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
    "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
    "numpy",
    "pandas",
    "joblib",
    "scikit-learn"
        
            ]
)
