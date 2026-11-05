PR17 head 3abe52d3f5c3746f44369c2eeb10e6c4c6190581; parent ca291a5c07a6098130cdae17b4eda37e9f3b7247. The only change from parent is torch.compile(mode="default") in solution.py, layered on the RoPE optimization. This test copy adds an empty models/__init__.py for Python package resolution; original files are preserved separately locally in submission/.
SHA256 relative to candidates/pr17_rope_3abe52:
6cd84f93be4e71f2c8856bb4d94d5ce3a3a62c008838ef335ae44e4ba512417e solution.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 models/__init__.py (added)
a791b73a71f9cfbd8f9821b726ae13aa0ccb3de9818f6bfecc0403d4532d9251 models/progen/__init__.py
a366655a755ebbb11861bcba58a7b3f719c0ec43d7c9a729a44869d17942bd7e models/progen/configuration_progen.py
7887f44ed116cdce67bdfff5be72086d5d68715d906eeeb65c3d01aa34115f9d models/progen/modeling_progen.py
Compilation is lazy: errors on the first model invocation are outside the try/except. The success print does not prove compilation completed. DCU compatibility, numerical correctness, compile overhead and dynamic shape behavior are untested. No model was run.
