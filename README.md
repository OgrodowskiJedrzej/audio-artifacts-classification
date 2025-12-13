<a id="readme-top"></a>

# Audio Artifacts Detection ML Project

[![Contributors][contributors-shield]][contributors-url]  
[![Forks][forks-shield]][forks-url]  
[![Stars][stars-shield]][stars-url]  
[![Issues][issues-shield]][issues-url]  
[![License][license-shield]][license-url]  

---

## Table of Contents

- [About The Project](#about-the-project)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## About The Project

This project prototypes a **machine learning model to detect audio artifacts** from hearing aid test measurements.  

Key highlights:  

- Trained on real production measurements and samples (private data) thus original dataset could not be published.  
- Azure ML was used to orchestrate training pipelines and GPU compute.  
- Models implemented in PyTorch with configurable hyperparameters.  
- Experiment tracking via MLflow.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started
### Prerequisites

- Ubuntu or macOS (not tested on Windows)  
- Python 3.12+  
- Azure ML stack

### Installation

```bash
# Clone the repository
git clone https://github.com/your_username/repo_name.git
cd repo_name

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Jedrzej Ogrodowski (JDOG)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/OgrodowskiJedrzej/audio-artifacts-classification.svg?style=for-the-badge
[contributors-url]: https://github.com/OgrodowskiJedrzej/audio-artifacts-classification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/OgrodowskiJedrzej/audio-artifacts-classification.svg?style=for-the-badge
[forks-url]: https://github.com/OgrodowskiJedrzej/audio-artifacts-classification/network/members
[stars-shield]: https://img.shields.io/github/stars/OgrodowskiJedrzej/audio-artifacts-classification.svg?style=for-the-badge
[stars-url]: https://github.com/OgrodowskiJedrzej/audio-artifacts-classification/stargazers
[issues-shield]: https://img.shields.io/github/issues/OgrodowskiJedrzej/audio-artifacts-classification.svg?style=for-the-badge
[issues-url]: https://github.com/OgrodowskiJedrzej/audio-artifacts-classification/issues
[license-shield]: https://img.shields.io/github/license/OgrodowskiJedrzej/audio-artifacts-classification.svg?style=for-the-badge
[license-url]: https://github.com/OgrodowskiJedrzej/audio-artifacts-classification/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png