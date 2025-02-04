def test_help_option(script_runner, monkeypatch):
    ret = script_runner.run('scil_labelseg.py', '--help')

    assert not ret.success
