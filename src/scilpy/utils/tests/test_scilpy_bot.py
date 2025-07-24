
from scilpy.utils.scilpy_bot import (
    _make_title, _get_docstring_from_script_path,
    _split_first_sentence, _stem_word, _stem_keywords, _stem_text,
    _stem_phrase, _highlight_keywords, _get_synonyms,
    _extract_keywords_and_phrases, _calculate_score
)


def test_make_title():
    result = _make_title("Test Title")
    assert "Test Title" in result


def test_get_docstring_from_script_path(tmp_path):
    script_content = '"""This is a test docstring."""'
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    result = _get_docstring_from_script_path(str(script_path))
    assert result == "This is a test docstring."


def test_split_first_sentence():
    text = "This is the first sentence. This is the second sentence."
    first, remaining = _split_first_sentence(text)
    assert first == "This is the first sentence."
    assert remaining == " This is the second sentence."


def test_stem_keywords():
    keywords = ["running", "jumps"]
    result = _stem_keywords(keywords)
    assert result == ["run", "jump"]


def test_stem_text():
    text = "Running and jumping."
    result = _stem_text(text)
    assert result == "run and jump ."


def test_stem_phrase():
    phrase = "Running quickly"
    result = _stem_phrase(phrase)
    assert result == "run quickli"


def test_highlight_keywords():
    text = "Running and jumping."
    stemmed_keywords = ["run"]
    result = _highlight_keywords(text, stemmed_keywords)
    assert "Running" in result


def test_get_synonyms():
    synonyms_data = [["run", "sprint"], ["jump", "leap"]]
    result = _get_synonyms("run", synonyms_data)
    # There is no synonyms for "run" in the provided data
    assert len(result) == 2


def test_extract_keywords_and_phrases():
    keywords = ["running", "jumps", "quick run"]
    result_keywords, result_phrases = _extract_keywords_and_phrases(keywords)

    # Verify all keywords and phrases are extracted
    assert len(result_keywords) == 2
    for expected in ["running", "jumps"]:
        assert expected in result_keywords

    assert len(result_phrases) == 1
    assert "quick run" in result_phrases


def test_stem_word_specific():
    result = _stem_word("streamlines")
    assert result == "streamlin"

    result = _stem_word("tractograms")
    assert result == "tractogram"

    result = _stem_word("tractography")
    assert result == "tractographi"

    result = _stem_word("tractometry")
    assert result == "tractometri"


def test_calculate_score():
    keywords = ["run"]
    phrases = ["quick run"]
    text = "Running quickly is fun. A quick run is good."
    filename = "run_script.py"
    result = _calculate_score(keywords, phrases, text, filename)
    assert result["run"] == 2
    assert result["quick run"] == 1
