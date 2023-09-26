from pathlib import Path

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="askcode",
    version="0.1.0",
    author="Abdeladim Sadiki",
    description="Chat with your code base",
    long_description=long_description,
    ext_modules=[],
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages('.'),
    package_dir={'': '.'},
    long_description_content_type="text/markdown",
    license='MIT',
    project_urls={
        'Documentation': 'https://github.com/abdeladim-s/askcode',
        'Source': 'https://github.com/abdeladim-s/askcode',
        'Tracker': 'https://github.com/abdeladim-s/askcode/issues',
    },
    install_requires=["langchain", "chromadb", "tiktoken", "openai", "rich", "auto_gptq", "transformers"],
    entry_points={
        'console_scripts': ['askcode=askcode.cli:run']
    },
)
