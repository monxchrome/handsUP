# HANDSUP AI

<img src="https://img.icons8.com/?size=512&id=kTuxVYRKeKEY&format=png" width="100" />

Detecting hand parts, controlling windows media controllers using gestures, and identifying road lanes using computer vision.

[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/eli64s/readme-ai/.github%2Fworkflows%2Frelease-pipeline.yml?logo=GitHub&label=handsup&color=c125ff)](https://github.com/monxchrome/handsUP)
[![PyPI version](https://img.shields.io/pypi/v/readmeai?color=c125ff)](https://pypi.org/project/pypi-latest/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/readmeai.svg?color=c125ff)](https://pypi.python.org/pypi/readmeai/)
![License: Apache](https://img.shields.io/github/license/eli64s/readme-ai?color=c125ff)

---

## 🔗 Quick Links
* [Overview](#-overview)
* [Features](#-features)
* [Getting Started](#-getting-started)
* [Project Roadmap](#-project-roadmap)
* [Contributing](#-contributing)
* [License](#-license)

---

## 🔭 Overview

***Objective***

<em>HANDSUP-AI</em> Our unique and innovative AI technologies can be trusted when it comes to interacting with your computer and the road environment. Our AI has a unique set of features that includes hand detection and gesture recognition to control Windows Media Player, as well as detecting road lines to intelligently change lanes.<br>

## 🤖 Features

<br>
<div>
<details>
    <summary style="display: flex; align-items: center;">
        <span style="font-size: 2.0em;"> ❶ Hand and gesture detection</span>
    </summary>
    <table>
        <tr>
            <td>
                <h4><i>Project Hand and gesture detection</i></h4>
                <p>
                    ‣ Our AI exhibits high precision in recognizing user hands and dynamic gestures, making Windows Media Player control more intuitive and convenient. Easily adjust volume, switch tracks, and manage multimedia functions with natural hand movements.
                </p>
            </td>
        </tr>
        <tr>
            <td>
                <img src="./public/1.png" alt="badges" />
            </td>
        </tr>
    </table>
</details>
</div>
<br>
<div>
    <details>
        <summary style="display: flex; align-items: center;">
            <span style="font-size: 2.0em;"> ❷ Lane Detection on Roads</span>
        </summary>
        <table>
            <tr>
                <td colspan="2">
                    <h4><i>Lane Detection on Roads</i></h4>
                </td>
            </tr>
            <tr>
                <td colspan="2">
                    <p>‣ Equipped with advanced computer vision algorithms, our AI accurately identifies road lines. This functionality is crucial for implementing automated lane-changing maneuvers. The AI automatically analyzes the surrounding environment, identifies available lanes, and makes decisions for safe lane changes in accordance with traffic rules.</p>
                </td>
            </tr>
            <tr>
                <td align="center">
                    <img src="./public/2.png" alt="repository-tree" />
                </td>
            </tr>
        </table>
    </details>
</div>
<br>
<div>
    <details>
        <summary style="display: flex; align-items: center;">
            <span style="font-size: 2.0em;"> ❸ Integration with Windows Media</span>
        </summary>
        <table>
            <tr>
                <td>
                    <h4><i>Integration with Windows Media</i></h4>
                    <p>
                        ‣ Our AI seamlessly integrates with Windows Media Player, ensuring smooth and efficient control of multimedia files. Simple gestures allow you to enjoy your music and videos effortlessly, making the control process more natural.
                    </p>
                </td>
            </tr>
            <tr>
                <td>
                    <img src="./public/3.png" alt="feature-table" />
                </td>
                <td>
                    <img src="./public/4.png" alt="feature-table" />
                </td>
            </tr>
        </table>
    </details>
</div>
<br>
<br>

<p align="right">
  <a href="#top"><b>Return </b></a>
</p>

---

## 👩‍💻 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

- *Python 3.9+*
- *Pip, Docker, Poetry, Pipenv, or Conda*

<br>

***Repository***

A remote repository URL or local directory path to your project is needed to use *handsup-ai*. The following platforms are currently supported:
- *GitHub*
- *GitLab*
- *Bitbucket*
- *File System*

<br>

---

### 📦 Installation

Using `pip`
```bash
pip install opencv-python, PyAutoGUI, mediapipe
```

Using `conda`
```bash
conda install -c conda-forge opencv-python, PyAutoGUI, mediapipe
```

<br>

Alternatively, clone the readme-ai repository and build from source.

```bash
git clone https://github.com/monxchrome/handsUP && \
cd handsUP
```

Using `pipenv`
```bash
pipenv install && \
pipenv shell
```

<br>

Using `poetry`
```bash
poetry install && \
poetry shell
```

---

### 🚀 Running *HANDSUP-AI*

Create environment.

On `Linux/MacOS`
```bash
python -m venv venv
source venv/bin/activate
```

On `Windows`
```bash
python -m venv venv
source venv/scripts/activate
```

Next, run app.

Using `python3`
```bash
python3 HandTracking.py
```

or

Using `python3`
```bash
python3 volume_control.py
```

or

Using `python3`
```bash
python3 Screen Brightness.py
```

<br>

Using `conda`
```bash
python HandTracking.py
```

or

Using `conda`
```bash
python volume_control.py
```

or

Using `conda`
```bash
python Screen Brightness.py
```

---

## 🛠 Project Roadmap

- [X] Publish project as a Python library via PyPI for easy installation.
  - [*PyPI - handsUP*]
- [X] Refactor our handsUP code.
- [ ] Building template system to create HANDSUP files for specific use-cases (data, mobile, web, etc.)
- [ ] Add support for generating HANDSUP files in any language (i.e. CN, ES, FR, JA, KO, RU).
- [ ] Develop GitHub Actions script to automatically update the HANDSUP file when new code is pushed.

---

## 📄 License

[Apache](https://github.com/monxchrome/handsUP/blob/main/LICENSE)

---

<p align="right">
  <a href="#top"><b>Return</b></a>
</p>