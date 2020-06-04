from setuptools import setup, find_packages


def setup_package():

    with open("readme.md", "r") as fh:
        long_description = fh.read()
    with open("requirements.txt") as fp:
        install_requires = fp.read().strip().split("\n")

    setup \
        (
            install_requires=install_requires,
            classifiers=[
                'Development Status :: 3 - Alpha',
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                "Operating System :: OS Independent",
            ],
            python_requires='>=3.7',
            entry_points=
            {
                "console_scripts":
                    [
                        "xehm = xehm.app.run:main"
                    ]
            }
        )


if __name__ == "__main__":
    setup_package()
