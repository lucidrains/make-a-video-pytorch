from setuptools import setup, find_packages

setup(
  name = 'make-a-video-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.4.0',
  license='MIT',
  description = 'Make-A-Video - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/make-a-video-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'text-to-video',
    'axial convolutions'
  ],
  install_requires=[
    'classifier-free-guidance-pytorch',
    'einops>=0.6',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
