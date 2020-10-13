from setuptools import setup
with open('requirements.txt','r') as f:
    requirements = f.readlines()

setup(
   name='mctsextend',
   version='1.0',
   description='Package for discovering interesting sequences',
   author='Blind',
   packages=['mctsextend'],
   install_requires=requirements,
   scripts=['xp/xp_main.py']
)
