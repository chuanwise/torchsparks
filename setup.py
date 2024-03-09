#  Copyright 2024 Chuanwise and contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import setuptools

with open("README.md", "r", encoding="utf-8") as file_descriptor:
    long_description = file_descriptor.read()

setuptools.setup(
    name="torchsparks",
    version="0.1.2",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
    ],
    author="Chuanwise",
    author_email="i@chuanwise.cn",
    description="Tools for PyTorch in Pure Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache Software License",
    url="https://github.com/chuanwise/torchsparks",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
    ],
)
