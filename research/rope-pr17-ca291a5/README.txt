PR #17, commit ca291a5c07a6098130cdae17b4eda37e9f3b7247
This test copy preserves the four original files and adds only an empty models/__init__.py to prevent Python resolving the official models package instead. Original snapshot is retained separately locally in submission/.
SHA256 (relative to candidates/pr17_rope_ca291a5):
8b81f3b6e7c65e7a484a827e4857841cbd0ca30b860a76ccd9ebb0498f263ffc  solution.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  models/__init__.py (added)
a791b73a71f9cfbd8f9821b726ae13aa0ccb3de9818f6bfecc0403d4532d9251  models/progen/__init__.py
a366655a755ebbb11861bcba58a7b3f719c0ec43d7c9a729a44869d17942bd7e  models/progen/configuration_progen.py
7887f44ed116cdce67bdfff5be72086d5d68715d906eeeb65c3d01aa34115f9d  models/progen/modeling_progen.py
The candidate computes sin/cos on CPU before moving buffers to the device; original computes sin/cos on device. Numerical agreement and device execution are unverified. No model or device test performed.
