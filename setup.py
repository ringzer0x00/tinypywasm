import os.path
import setuptools

root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name='tinypywasm',
    version='1.0.0',
    url='https://github.com/ringzer0x00/tinypywasm',
    license='MIT',
    author='ringzer0x00',
    author_email='mattia.paccamiccio@unicam.it',
    description='TinyWASM Interpreter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['tinypywasm'],
    python_requires='>=3.6',
    install_requires=['numpy'],
)
